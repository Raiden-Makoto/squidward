---
name: update-image
description: Update the dsv4 container on a remote box to a new rocm/sgl-dev image, preserving the container's run config. Do NOT start the server.
---

# Update dsv4 Container Image

Use this to move a box's `dsv4` container to a new `rocm/sgl-dev:<tag>` image. Match the image
arch to the box (`mi30x` = gfx942 dsv4 box; `mi35x` = gfx950 gbt box — see `known-boxes.mdc`).

## Rules
- Recreate the container only; **do NOT start the sglang server** afterward (leave it for the
  user to launch).
- Touch only the `dsv4` container. Never disturb other tenants' containers on the box.
- Follow `remote-box-rules.mdc` (no upstream edits, no pip install, box never ahead of local).

## Steps
1. **Pull** the new image on the box:
   `docker pull rocm/sgl-dev:<new-tag>`
2. **Capture** the existing container's run config so the recreate is faithful:
   `docker inspect dsv4` → record Devices, Mounts (Source:Destination), Env (esp. `HF_HOME`),
   GroupAdd, CapAdd, IpcMode, ShmSize, NetworkMode, SecurityOpt.
3. **Stop + remove** the old container:
   `docker stop dsv4 && docker rm dsv4`
4. **Recreate** with the new image, preserving the captured config. Canonical form (adjust
   mounts/HF_HOME to what step 2 reported for that box):
   ```bash
   docker run -d --name dsv4 \
     --device=/dev/kfd --device=/dev/dri \
     --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
     --group-add video --group-add render \
     --ipc=host --shm-size=64g --network=host \
     -v <hf_home_src>:/root/hf_home -e HF_HOME=/root/hf_home \
     rocm/sgl-dev:<new-tag> sleep infinity
   ```
5. **Verify**: `docker ps --filter name=dsv4` shows the new image and `Up`. Confirm the fork at
   `/sgl-workspace/squidward` is present (re-mount/clone if the box mounts it from host) and at
   the expected commit.
6. **Do not start the server.** Report the new image tag and container status.
