use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    buffer::{BufferUsage, CpuAccessibleBuffer, BufferContents, cpu_access::ReadLock, DeviceLocalBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::PhysicalDeviceType,
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, Queue,
    },
    instance::{Instance, InstanceCreateInfo},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture}, shader::ShaderModule, memory::pool::{PotentialDedicatedAllocation, StandardMemoryPoolAlloc},
};

#[derive(Debug)]
pub struct ComputeDevice {
    device: Arc<Device>,
    pipeline: Arc<ComputePipeline>,
    queue: Arc<Queue>
}

impl ComputeDevice {
    pub fn new<F: FnOnce(Arc<Device>) -> Arc<ShaderModule>>(glsl_src_factory: F) -> Self {
        let library = VulkanLibrary::new().unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
                enumerate_portability: true,
                ..Default::default()
            },
        )
        .unwrap();
    
        // Choose which physical device to use.
        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
                // that supports compute operations.
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.compute)
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();
    
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );
    
        // Now initializing the device.
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();
    
        // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
        // example we use only one queue, so we just retrieve the first and only element of the
        // iterator and throw it away.
        let queue = queues.next().unwrap();

        // Now let's get to the actual example.
        //
        // What we are going to do is very basic: we are going to fill a buffer with 64k integers
        // and ask the GPU to multiply each of them by 12.
        //
        // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
        // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
        // or four by four, a GPU will do it by groups of 32 or 64.
        //
        // Note however that in a real-life situation for such a simple operation the cost of
        // accessing memory usually outweighs the benefits of a faster calculation. Since both the CPU
        // and the GPU will need to access data, there is no other choice but to transfer the data
        // through the slow PCI express bus.

        // We need to create the compute pipeline that describes our operation.
        //
        // If you are familiar with graphics pipeline, the principle is the same except that compute
        // pipelines are much simpler to create.



        let pipeline = {
            ComputePipeline::new(
                device.clone(),
                glsl_src_factory(device.clone()).entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };

        Self { device: device, pipeline: pipeline, queue: queue }

    }


    pub fn compute<T, II: IntoIterator<Item = T>, R, F: FnOnce(ReadLock<'_, [T], PotentialDedicatedAllocation<StandardMemoryPoolAlloc>>) -> R>(
        &self, 
        dims: [u32; 3], 
        input_data: II, 
        input_data_count: usize,
        r: F
    ) -> R
    where 
        [T]: BufferContents,
        <II as IntoIterator>::IntoIter: ExactSizeIterator {


        let accessible_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage { transfer_src: true, transfer_dst: true, ..Default::default() },
            false,
            input_data.into_iter(),
        )
        .unwrap();
     
         // Create a buffer array on the GPU with enough space for `10_000` floats.
        let device_local_buffer = DeviceLocalBuffer::<[T]>::array(
            self.device.clone(),
            input_data_count as u64,
            BufferUsage { storage_buffer: true, transfer_dst: true, transfer_src: true, ..Default::default() },
            self.device.active_queue_family_indices().iter().map(|a|*a),
        )
        .unwrap();
     
         // Create a one-time command to copy between the buffers.
        let mut cbb = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.copy_buffer(CopyBufferInfo::buffers(
            accessible_buffer.clone(),
            device_local_buffer.clone(),
        ))
         .unwrap();
         let cb = cbb.build().unwrap();
     
         // Execute copy command and wait for completion before proceeding.
        cb.execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        // In order to let the shader access the buffer, we need to build a *descriptor set* that
        // contains the buffer.
        //
        // The resources that we bind to the descriptor set must match the resources expected by the
        // pipeline which we pass as the first parameter.
        //
        // If you want to run the pipeline on multiple different buffers, you need to create multiple
        // descriptor sets that each contain the buffer you want to run the shader on.
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, device_local_buffer.clone())
                ],
        )
        .unwrap();

        // In order to execute our operation, we have to build a command buffer.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        
        builder
            // The command buffer only does one thing: execute the compute pipeline.
            // This is called a *dispatch* operation.
            //
            // Note that we clone the pipeline and the set. Since they are both wrapped around an
            // `Arc`, this only clones the `Arc` and not the whole pipeline or set (which aren't
            // cloneable anyway). In this example we would avoid cloning them since this is the last
            // time we use them, but in a real code you would probably need to clone them.
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch(dims)
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(device_local_buffer, accessible_buffer.clone()))
            .unwrap();
        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        // Let's execute this command buffer now.
        // To do so, we TODO: this is a bit clumsy, probably needs a shortcut
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            // This line instructs the GPU to signal a *fence* once the command buffer has finished
            // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
            // reached a certain point.
            // We need to signal a fence here because below we want to block the CPU until the GPU has
            // reached that point in the execution.
            .then_signal_fence_and_flush()
            .unwrap();

        // Blocks execution until the GPU has finished the operation. This method only exists on the
        // future that corresponds to a signalled fence. In other words, this method wouldn't be
        // available if we didn't call `.then_signal_fence_and_flush()` earlier.
        // The `None` parameter is an optional timeout.
        //
        // Note however that dropping the `future` variable (with `drop(future)` for example) would
        // block execution as well, and this would be the case even if we didn't call
        // `.then_signal_fence_and_flush()`.
        // Therefore the actual point of calling `.then_signal_fence_and_flush()` and `.wait()` is to
        // make things more explicit. In the future, if the Rust language gets linear types vulkano may
        // get modified so that only fence-signalled futures can get destroyed like this.
        future.wait(None).unwrap();

        // Now that the GPU is done, the content of the buffer should have been modified. Let's
        // check it out.
        // The call to `read()` would return an error if the buffer was still in use by the GPU.

        r(accessible_buffer.read().unwrap())
    }
}


