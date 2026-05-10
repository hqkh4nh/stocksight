# Import this before `import tensorflow` to silence TF logs and enable GPU memory growth.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

try:
    import tensorflow as tf
    for _gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(_gpu, True)
        except RuntimeError:
            # Already initialized; env var above takes effect on next process.
            pass
except Exception:
    pass