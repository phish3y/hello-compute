use bytemuck::{Pod, Zeroable};
use wgpu::{PollType, util::DeviceExt};

use super::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Point {
    x: f32,
    y: f32,
}

impl GpuContext {
    pub async fn haversine(&self, a: &[geo::Point<f32>], b: &[geo::Point<f32>]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "input arrays must have the same length");

        let a_gpu: Vec<Point> = a.iter().map(|p| Point { x: p.x(), y: p.y() }).collect();
        let b_gpu: Vec<Point> = b.iter().map(|p| Point { x: p.x(), y: p.y() }).collect();

        let shader = self
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/haversine.wgsl"));

        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a buffer"),
                contents: bytemuck::cast_slice(&a_gpu),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b buffer"),
                contents: bytemuck::cast_slice(&b_gpu),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (a.len() * std::mem::size_of::<f32>()) as u64;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Haversine Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Haversine Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Encoder"),
            });

        let workgroups = (a.len() as f32 / 64.0).ceil() as u32;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        let readback_slice = readback_buf.slice(..);
        readback_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(PollType::Wait).unwrap();

        let data = readback_slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{HaversineDistance, Point};

    #[test]
    fn test_haversine_gpu_matches_cpu() {
        let ctx = pollster::block_on(GpuContext::new()).unwrap();

        let a = vec![
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(10.0, 10.0),
        ];
        let b = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 0.0),
            Point::new(10.0, 14.0),
        ];

        let gpu_result = pollster::block_on(ctx.haversine(&a, &b));

        let expected: Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(p1, p2)| p1.haversine_distance(p2) as f32)
            .collect();

        for (i, (g, e)) in gpu_result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.5,
                "mismatch at index {i}: GPU result = {g}, CPU expected = {e}, diff = {}",
                (g - e).abs()
            );
        }
    }
}
