use wgpu::{util::DeviceExt, PollType};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PointData {
    x: f32,
    y: f32,
    value: f32,
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let points = vec![
        PointData { x: 1.0, y: 2.0, value: 10.0 },
        PointData { x: 3.0, y: 4.0, value: 20.0 },
        PointData { x: 5.0, y: 6.0, value: 30.0 },
    ];

    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Staging Buffer"),
        contents: bytemuck::cast_slice(&points),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_size = (points.len() * std::mem::size_of::<PointData>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: Some("Bind Group Layout"),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: staging_buffer.as_entire_binding(),
        }],
        label: Some("Bind Group"),
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader/compute.wgsl"));
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(((points.len() as f32) / 64.0).ceil() as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&staging_buffer, 0, &output_buffer, 0, buffer_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
    device.poll(PollType::Wait).unwrap();
    let data = buffer_slice.get_mapped_range();

    let result: &[PointData] = bytemuck::cast_slice(&data);
    println!("Modified Points:");
    for p in result {
        println!("{:?}", p);
    }
}