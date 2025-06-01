use core::fmt;

pub mod functions;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new() -> Result<Self, GpuContextError> {
        let instance: wgpu::Instance = wgpu::Instance::default();
        let adapter: wgpu::Adapter = instance
            .request_adapter(&Default::default())
            .await
            .map_err(|err| GpuContextError {
                message: format!("failed to request adapter: {}", err),
            })?;

        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .map_err(|err| GpuContextError {
                message: format!("failed to request device: {}", err),
            })?;

        Ok(GpuContext { device, queue })
    }
}

#[derive(Debug)]
pub struct GpuContextError {
    pub message: String,
}

impl fmt::Display for GpuContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuContextError: {}", self.message)
    }
}

impl std::error::Error for GpuContextError {}
