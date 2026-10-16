# Sourced inside apptainer after `conda activate ccsynergy`.
# Keeps host NVIDIA driver libs (apptainer --nv → /.singularity.d/libs) first,
# then tensorflow[and-cuda] nvidia-* wheel libs, then conda libs.

# shellcheck disable=SC2155
if [ -d /.singularity.d/libs ]; then
  export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

_CCSY_SITE="/opt/conda/envs/ccsynergy/lib/python3.10/site-packages"
if [ -d "${_CCSY_SITE}/nvidia" ]; then
  for _nvlib in "${_CCSY_SITE}"/nvidia/*/lib "${_CCSY_SITE}"/nvidia/*/lib64; do
    [ -d "${_nvlib}" ] && LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_nvlib}"
  done
  export LD_LIBRARY_PATH
fi
unset _CCSY_SITE _nvlib

# Fail fast unless ALLOW_CPU=1
if [ "${ALLOW_CPU:-0}" != "1" ]; then
  python - <<'PY'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", gpus)
if not gpus:
    print(
        "ERROR: no GPU visible to TensorFlow. Rebuild ccsynergy.sif with "
        "tensorflow[and-cuda] (see containers/ccsynergy.def) and run with "
        "apptainer --nv. Or set ALLOW_CPU=1 to proceed on CPU.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
fi
