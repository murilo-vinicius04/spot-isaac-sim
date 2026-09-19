# Isaac Sim 6.0 for spot_isaac6/ (based on EESC-LabRoM/spot-teleop Dockerfile.isaac).
# The sim runners use Isaac 6 APIs (SimulationManager/SimulationEvent, PhysxJointAxisAPI friction efforts), so 5.x will not work.
FROM nvcr.io/nvidia/isaac-sim:6.0.0

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1

USER root

# The Spot policy is an ONNX file; Isaac's Python does not ship onnxruntime. --no-deps keeps Isaac's own numpy/protobuf.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps onnxruntime==1.23.2

WORKDIR /repo
