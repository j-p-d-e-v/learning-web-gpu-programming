# Learning WGPU

Resources:
- https://sotrh.github.io/learn-wgpu/beginner/tutorial5-textures/#getting-data-into-a-texture
- https://docs.rs/wgpu/latest/wgpu/

# Building the WASM
```sh
wasm-pack build -d .\wgpu\pkg\ -t web
```

# UI Environement
```sh
docker-compose up -d
docker exec -it <container_id> "bash"
cd wgpu
yarn dev
```

Open browser:
```
http://localhost:7080/
```