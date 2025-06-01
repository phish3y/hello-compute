use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use wgpu::{PollType, util::DeviceExt};

use super::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BBox {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl From<&geo::Rect<f32>> for BBox {
    fn from(rect: &geo::Rect<f32>) -> Self {
        let min = rect.min();
        let max = rect.max();
        BBox {
            min_x: min.x,
            min_y: min.y,
            max_x: max.x,
            max_y: max.y,
        }
    }
}

impl From<BBox> for geo::Rect<f32> {
    fn from(b: BBox) -> Self {
        geo::Rect::new(
            geo::Coord {
                x: b.min_x,
                y: b.min_y,
            },
            geo::Coord {
                x: b.max_x,
                y: b.max_y,
            },
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Pair {
    left_index: u32,
    right_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Counter {
    count: u32,
    _pad: [u32; 3],
}

impl GpuContext {
    pub async fn filter_intersecting(
        &self,
        left: Vec<geo::Rect<f32>>,
        right: Vec<geo::Rect<f32>>,
    ) -> Vec<geo::Rect<f32>> {
        let left_gpu: Vec<BBox> = left.iter().map(BBox::from).collect();
        let right_gpu: Vec<BBox> = right.iter().map(BBox::from).collect();

        let shader = self
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/filter_intersecting.wgsl"));

        let left_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Left Buffer"),
                contents: bytemuck::cast_slice(&left_gpu),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let right_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Right Buffer"),
                contents: bytemuck::cast_slice(&right_gpu),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (left.len() * right.len() * std::mem::size_of::<Pair>()) as u64;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pair_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counter_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Counter Buffer"),
                contents: bytemuck::bytes_of(&Counter {
                    count: 0,
                    _pad: [0; 3],
                }),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let counter_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counter Readback"),
            size: std::mem::size_of::<Counter>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Filter Intersecting Bind Group Layout"),
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counter_buf.as_entire_binding(),
                },
            ],
            label: Some("Bind Group"),
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
                label: Some("Compute Pipeline"),
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

        let workgroups = ((left.len() * right.len()) as f32 / 64.0).ceil() as u32;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &pair_readback, 0, output_size);
        encoder.copy_buffer_to_buffer(
            &counter_buf,
            0,
            &counter_readback,
            0,
            std::mem::size_of::<Counter>() as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let mut seen = HashSet::new();
        let mut filtered = Vec::new();
        for pair in get_pairs(&self.device, &counter_readback, &pair_readback) {
            if seen.insert(pair.left_index) {
                if let Some(bbox) = left.get(pair.left_index as usize) {
                    filtered.push(*bbox);
                }
            }
        }

        filtered
    }
}

fn get_pairs(
    device: &wgpu::Device,
    counter_readback: &wgpu::Buffer,
    pair_readback: &wgpu::Buffer,
) -> Vec<Pair> {
    // Get the count of pairs
    let count_slice = counter_readback.slice(..);
    count_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(PollType::Wait).unwrap();

    let count_data = count_slice.get_mapped_range();
    let counter: &Counter = bytemuck::from_bytes(&count_data);
    let valid_len = counter.count as usize;

    // Use the count of pairs to get the pairs
    let pair_slice = pair_readback.slice(..(valid_len * std::mem::size_of::<Pair>()) as u64);
    pair_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(PollType::Wait).unwrap();

    let pair_data = pair_slice.get_mapped_range();
    bytemuck::cast_slice(&pair_data).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_intersecting_simple() {
        let ctx = pollster::block_on(GpuContext::new()).unwrap();

        let left = vec![
            geo::Rect::new(geo::Coord { x: 0.0, y: 0.0 }, geo::Coord { x: 2.0, y: 2.0 }),
            geo::Rect::new(geo::Coord { x: 3.0, y: 3.0 }, geo::Coord { x: 5.0, y: 5.0 }),
            geo::Rect::new(
                geo::Coord { x: 9.0, y: 9.0 },
                geo::Coord { x: 11.0, y: 11.0 },
            ),
        ];

        let right = vec![
            geo::Rect::new(geo::Coord { x: 1.0, y: 1.0 }, geo::Coord { x: 4.0, y: 4.0 }),
            geo::Rect::new(geo::Coord { x: 6.0, y: 6.0 }, geo::Coord { x: 8.0, y: 8.0 }),
        ];

        let filtered = pollster::block_on(ctx.filter_intersecting(left.clone(), right.clone()));

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&left[0]));
        assert!(filtered.contains(&left[1]));
    }
}
