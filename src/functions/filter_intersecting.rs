use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use wgpu::{PollType, util::DeviceExt};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BBox {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
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

pub async fn filter_intersecting(left: Vec<BBox>, right: Vec<BBox>) -> Vec<BBox> {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/filter_intersecting.wgsl"));

    let left_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Left Buffer"),
        contents: bytemuck::cast_slice(&left),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let right_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Right Buffer"),
        contents: bytemuck::cast_slice(&right),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let max_pairs = (left.len() * right.len()) as u64;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: max_pairs * std::mem::size_of::<Pair>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: max_pairs * std::mem::size_of::<Pair>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Counter Buffer"),
        contents: bytemuck::bytes_of(&Counter {
            count: 0,
            _pad: [0; 3],
        }),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let counter_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Counter Readback"),
        size: std::mem::size_of::<Counter>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        label: Some("Bind Group Layout"),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

    encoder.copy_buffer_to_buffer(
        &output_buf,
        0,
        &readback_buf,
        0,
        max_pairs * std::mem::size_of::<Pair>() as u64,
    );
    encoder.copy_buffer_to_buffer(
        &counter_buf,
        0,
        &counter_readback,
        0,
        std::mem::size_of::<Counter>() as u64,
    );
    queue.submit(Some(encoder.finish()));

    let count_slice = counter_readback.slice(..);
    count_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(PollType::Wait).unwrap();

    let count_data = count_slice.get_mapped_range();
    let counter: &Counter = bytemuck::from_bytes(&count_data);
    let valid_len = counter.count as usize;

    let pair_slice = readback_buf.slice(..(valid_len * std::mem::size_of::<Pair>()) as u64);
    pair_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(PollType::Wait).unwrap();

    let pair_data = pair_slice.get_mapped_range();
    let pairs: &[Pair] = bytemuck::cast_slice(&pair_data);

    let mut seen = HashSet::new();
    let mut filtered = Vec::new();
    for pair in pairs {
        if seen.insert(pair.left_index) {
            if let Some(bbox) = left.get(pair.left_index as usize) {
                filtered.push(*bbox);
            }
        }
    }

    filtered
}
