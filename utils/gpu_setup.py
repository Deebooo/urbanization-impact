import tensorflow as tf

def setup_gpus(memory_limit=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            gpu_index = 0  # This is usually 0 for a single GPU setup
            if memory_limit:
                # Set up a virtual GPU with the specified memory limit
                config = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)
                tf.config.experimental.set_virtual_device_configuration(gpus[gpu_index], [config])
            else:
                # Allow TensorFlow to use as much GPU memory as needed
                tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)