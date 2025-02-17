import { mat4, vec3 } from "wgpu-matrix";
import { verticesArray } from "./models/grid3";
import Renderer from "./renderer";

let GRID_SIZE = 4;
const WORKGROUP_SIZE = 4;
const MAX_GRID_SIZE = 64;
const CELL_STATE_BUFFER_SIZE = MAX_GRID_SIZE ** 3 * 4; // 4 bytes per uint32

let step = 0;

const initialState = new Uint32Array(MAX_GRID_SIZE ** 3);

const setInitialState = (newGridSize: number) => {
  const relevantRangeEnd = newGridSize ** 3;
  for (let i = 0; i < relevantRangeEnd; ++i) {
    initialState[i] = Math.random() > 0.0 ? 1 : 0;
  }
};

setInitialState(GRID_SIZE);

const canvas = document.querySelector<HTMLCanvasElement>("canvas")!;

const restartButton =
  document.querySelector<HTMLButtonElement>("#restart-button");
restartButton?.addEventListener("click", async (_event) => {
  await device.queue.onSubmittedWorkDone();

  step = 0;
  device.queue.writeBuffer(cellStateStorage[0], 0, initialState);
  device.queue.writeBuffer(cellStateStorage[1], 0, initialState);
});

const gridSizeSelect = document.querySelector<HTMLSelectElement>("#grid-size");
gridSizeSelect?.addEventListener("change", async (event) => {
  const newSize = parseInt((event.target as HTMLSelectElement).value);
  await handleGridSizeChange(newSize);
});

const handleGridSizeChange = async (newSize: number) => {
  await device.queue.onSubmittedWorkDone();

  setInitialState(newSize);

  const newGridArray = new Float32Array([newSize, newSize, newSize]);
  device.queue.writeBuffer(gridBuffer, 0, newGridArray);

  // prettier-ignore
  device.queue.writeBuffer(cellStateStorage[0], 0, initialState, 0, newSize ** 3);
  // prettier-ignore
  device.queue.writeBuffer(cellStateStorage[1], 0, initialState, 0, newSize ** 3);

  step = 0;
  GRID_SIZE = newSize;
};

const { ctx, device } = await Renderer.create(canvas);

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({
  device: device,
  format: canvasFormat,
});

const gridArray = new Float32Array([GRID_SIZE, GRID_SIZE, GRID_SIZE]);
const gridBuffer = device.createBuffer({
  label: "Grid Uniforms",
  size: gridArray.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(gridBuffer, 0, gridArray);

const vertexBuffer = device.createBuffer({
  label: "Cell vertices",
  size: verticesArray.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, verticesArray);

const vertexBufferLayout: GPUVertexBufferLayout = {
  arrayStride: Float32Array.BYTES_PER_ELEMENT * 4,
  attributes: [
    {
      format: "float32x3",
      offset: 0,
      shaderLocation: 0,
    },
    {
      format: "float32",
      offset: Float32Array.BYTES_PER_ELEMENT * 3,
      shaderLocation: 1,
    },
  ],
};

const cellStateStorage = [
  device.createBuffer({
    label: "Cell State",
    size: CELL_STATE_BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
  device.createBuffer({
    label: "Cell State B",
    size: CELL_STATE_BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
];

device.queue.writeBuffer(cellStateStorage[0], 0, initialState);

const aspect = canvas.width / canvas.height;
const projectionMatrix = mat4.perspective((2 * Math.PI) / 6, aspect, 1, 100);
const viewMatrix = mat4.identity();
mat4.translate(viewMatrix, vec3.fromValues(0, 0, -4), viewMatrix);

const uniformMatrixBuffer = device.createBuffer({
  size: 3 * 4 * 16,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const matrixData = new Float32Array(32);
matrixData.set(viewMatrix, 0);
matrixData.set(projectionMatrix, 16);

device.queue.writeBuffer(uniformMatrixBuffer, 0, matrixData);

const getRotationMatrix = () => {
  const now = Date.now() / 1000;
  const rotationMatrix = mat4.identity();
  mat4.rotate(
    rotationMatrix,
    vec3.fromValues(Math.sin(now * 0.5), Math.cos(now * 0.5), 0),
    1,
    rotationMatrix,
  );

  return rotationMatrix;
};

const cellShaderModule = device.createShaderModule({
  label: "Cell shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec3f;
    @group(0) @binding(1) var<storage> cellState: array<u32>;
    @group(0) @binding(3) var<uniform> matrixData: MatrixData;

    struct MatrixData {
      view: mat4x4f,
      projection: mat4x4f,
      rotation: mat4x4f,
    };

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) encodedModelSpaceNormal: f32,
      @builtin(instance_index) instance: u32,
    };
    
    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) normal: vec3f,
      @location(1) worldPos: vec3f,
    };

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      let i = f32(input.instance);
      let cell = vec3f(
        i % grid.x,
        floor(i / grid.x % grid.y),
        floor(i / (grid.x * grid.y))
      );
      let state = f32(cellState[input.instance]);
      let cellOffset = (cell * 2.0 - grid + 1.0) / grid;
      let cubeSize = 1.0 / grid * 0.8;
      
      let position = vec4f(input.pos * cubeSize * state, 1.0);

      let modelMatrix = mat4x4f(
        vec4f(1, 0, 0, 0),
        vec4f(0, 1, 0, 0),
        vec4f(0, 0, 1, 0),
        vec4f(cellOffset, 1)
      );

      var output: VertexOutput;
      let rotatedModelPosition = matrixData.rotation * modelMatrix * position;
      output.pos = matrixData.projection * matrixData.view * rotatedModelPosition;

      var modelNormal: vec3f;
      switch u32(input.encodedModelSpaceNormal) {
        case 1:   { modelNormal  = vec3f( 0, -1,  0); }  // bottom
        case 2:   { modelNormal  = vec3f( 1,  0,  0); }  // right
        case 3:   { modelNormal  = vec3f( 0,  1,  0); }  // top
        case 4:   { modelNormal  = vec3f(-1,  0,  0); }  // left
        case 5:   { modelNormal  = vec3f( 0,  0,  1); }  // back
        default:  { modelNormal  = vec3f( 0,  0, -1); }  // front
      }
      output.normal = normalize((matrixData.rotation * vec4f(modelNormal, 0.0)).xyz);
      output.worldPos = rotatedModelPosition.xyz;

      return output;
    }

    struct FragInput {
      @location(0) normal: vec3f,
      @location(1) worldPos: vec3f,
    };

    @fragment
    fn fragmentMain(input: FragInput) -> @location(0) vec4f {
      let lightPos = vec3f(1.0, 1.0, 2.0);
      let lightDir = normalize(lightPos - input.worldPos);
      let diffuse = max(dot(input.normal, lightDir), 0.0);
      let ambient = 0.4;
      let color = vec3f(0.9, 0.9, 0.9) * clamp((diffuse + ambient), 0.0, 1.0);
      return vec4f(color, 1);
    }
  `,
});

const simulationShaderModule = device.createShaderModule({
  label: "Game of Life simulation shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec3f;
    @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

    fn cellIndex(cell: vec3u) -> u32 {
      return cell.x + cell.y * u32(grid.x) + cell.z * u32(grid.x * grid.y);
    }

    fn cellActive(x: u32, y: u32, z: u32) -> u32 {
      return cellStateIn[cellIndex(vec3u(x, y, z))];
    }

    @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
      var activeNeighbors: u32 = 0;
      
      for (var dx: u32 = 0; dx < 3; dx++) {
        for (var dy: u32 = 0; dy < 3; dy++) {
          for (var dz: u32 = 0; dz < 3; dz++) {
            if (dx == 1 && dy == 1 && dz == 1) {
              continue;
            }
            let nx = cell.x + dx - 1;
            let ny = cell.y + dy - 1;
            let nz = cell.z + dz - 1;
            activeNeighbors += cellActive(nx, ny, nz);
          }
        }
      }
      
      let i = cellIndex(cell.xyz);
      switch activeNeighbors {
        case 5: {
          cellStateOut[i] = 1;
        }
        case 1, 2, 3, 4, 8 {
          cellStateOut[i] = 0;
        }
        default: {
          cellStateOut[i] = cellStateIn[i];
        }
      }
    }
  `,
});

// Create the bind group layout and pipeline layout.
const bindGroupLayout = device.createBindGroupLayout({
  label: "Cell Bind Group Layout",
  entries: [
    {
      binding: 0,
      visibility:
        GPUShaderStage.VERTEX |
        GPUShaderStage.COMPUTE |
        GPUShaderStage.FRAGMENT,
      buffer: { type: "uniform" },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" }, // Cell state input buffer
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }, // Cell state output buffer
    },
    {
      binding: 3,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: "uniform" },
    },
  ],
});

const bindGroups = [
  device.createBindGroup({
    label: "Cell renderer bind group A",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: cellStateStorage[0] } },
      { binding: 2, resource: { buffer: cellStateStorage[1] } },
      { binding: 3, resource: { buffer: uniformMatrixBuffer } },
    ],
  }),
  device.createBindGroup({
    label: "Cell renderer bind group B",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: cellStateStorage[1] } },
      { binding: 2, resource: { buffer: cellStateStorage[0] } },
      { binding: 3, resource: { buffer: uniformMatrixBuffer } },
    ],
  }),
];

const pipelineLayout = device.createPipelineLayout({
  label: "Cell Pipeline Layout",
  bindGroupLayouts: [bindGroupLayout],
});

const cellPipeline = device.createRenderPipeline({
  label: "Cell pipeline",
  layout: pipelineLayout,
  vertex: {
    module: cellShaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout],
  },
  fragment: {
    module: cellShaderModule,
    entryPoint: "fragmentMain",
    targets: [
      {
        format: canvasFormat,
      },
    ],
  },
  primitive: {
    topology: "triangle-list",
    cullMode: "back",
  },
  depthStencil: {
    depthWriteEnabled: true,
    depthCompare: "less",
    format: "depth24plus",
  },
});

const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

// Create a compute pipeline that updates the game state.
const simulationPipeline = device.createComputePipeline({
  label: "Simulation pipeline",
  layout: pipelineLayout,
  compute: {
    module: simulationShaderModule,
    entryPoint: "computeMain",
  },
});

const updateGrid = () => {
  const encoder = device.createCommandEncoder();

  const computePass = encoder.beginComputePass();

  computePass.setPipeline(simulationPipeline);
  computePass.setBindGroup(0, bindGroups[step % 2]);

  const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(
    workgroupCount,
    workgroupCount,
    workgroupCount,
  );

  computePass.end();

  step++;

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: ctx.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0.3, a: 1 },
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });
  const rotationMatrix = getRotationMatrix();
  device.queue.writeBuffer(
    uniformMatrixBuffer,
    128,
    rotationMatrix.buffer,
    rotationMatrix.byteOffset,
    rotationMatrix.byteLength,
  );

  renderPass.setPipeline(cellPipeline);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.setBindGroup(0, bindGroups[step % 2]);

  renderPass.draw(verticesArray.length / 4, GRID_SIZE ** 3);

  renderPass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(updateGrid);
};

requestAnimationFrame(updateGrid);
