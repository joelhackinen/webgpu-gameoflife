import { mat4, vec3 } from "wgpu-matrix";
import { verticesArray } from "./models/grid3";
import Renderer from "./renderer";

let GRID_SIZE = 32;
const WORKGROUP_SIZE = 8;

let step = 0;
let currentInitialState: Uint32Array;

const initialStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
for (let i = 0; i < initialStateArray.length; ++i) {
  initialStateArray[i] = Math.random() > 0.6 ? 1 : 0;
}
currentInitialState = initialStateArray;

const canvas = document.querySelector<HTMLCanvasElement>("canvas")!;

const restartButton =
  document.querySelector<HTMLButtonElement>("#restart-button");
restartButton?.addEventListener("click", async (_event) => {
  await device.queue.onSubmittedWorkDone();

  step = 0;
  device.queue.writeBuffer(cellStateStorage[0], 0, currentInitialState);
  device.queue.writeBuffer(cellStateStorage[1], 0, currentInitialState);
});

const gridSizeSelect = document.querySelector<HTMLSelectElement>("#grid-size");
gridSizeSelect?.addEventListener("change", async (event) => {
  const newSize = parseInt((event.target as HTMLSelectElement).value);
  await handleGridSizeChange(newSize);
});

const handleGridSizeChange = async (newSize: number) => {
  await device.queue.onSubmittedWorkDone();

  GRID_SIZE = newSize;

  const newInitialStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
  for (let i = 0; i < newInitialStateArray.length; ++i) {
    newInitialStateArray[i] = Math.random() > 0.6 ? 1 : 0;
  }
  currentInitialState = newInitialStateArray;

  const newGridArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
  device.queue.writeBuffer(gridBuffer, 0, newGridArray);

  cellStateStorage[0].destroy();
  cellStateStorage[1].destroy();

  cellStateStorage[0] = device.createBuffer({
    label: "Cell State",
    size: newInitialStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  cellStateStorage[1] = device.createBuffer({
    label: "Cell State B",
    size: newInitialStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(cellStateStorage[0], 0, newInitialStateArray);
  device.queue.writeBuffer(cellStateStorage[1], 0, newInitialStateArray);

  bindGroups[0] = device.createBindGroup({
    label: "Cell renderer bind group A",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: cellStateStorage[0] } },
      { binding: 2, resource: { buffer: cellStateStorage[1] } },
      { binding: 3, resource: { buffer: uniformMatrixBuffer } },
    ],
  });
  bindGroups[1] = device.createBindGroup({
    label: "Cell renderer bind group B",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: cellStateStorage[1] } },
      { binding: 2, resource: { buffer: cellStateStorage[0] } },
      { binding: 3, resource: { buffer: uniformMatrixBuffer } },
    ],
  });

  step = 0;
};

const { ctx, device } = await Renderer.create(canvas);

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({
  device: device,
  format: canvasFormat,
});

const gridArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
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
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/ 0, verticesArray);

const vertexBufferLayout: GPUVertexBufferLayout = {
  arrayStride: 16,
  attributes: [
    {
      format: "float32x4",
      offset: 0,
      shaderLocation: 0,
    },
  ],
};

const cellStateStorage = [
  device.createBuffer({
    label: "Cell State",
    size: initialStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
  device.createBuffer({
    label: "Cell State B",
    size: initialStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
];

device.queue.writeBuffer(cellStateStorage[0], 0, initialStateArray);

const aspect = canvas.width / canvas.height;
const projectionMatrix = mat4.perspective((2 * Math.PI) / 9, aspect, 1, 100.0);
const modelViewProjectionMatrix = mat4.create();

const uniformMatrixBuffer = device.createBuffer({
  size: 4 * 16,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const getTransformationMatrix = () => {
  const viewMatrix = mat4.identity();
  mat4.translate(viewMatrix, vec3.fromValues(0, 0, -4), viewMatrix);
  const now = Date.now() / 1000;
  mat4.rotate(
    viewMatrix,
    vec3.fromValues(Math.sin(now * 0.5), Math.cos(now * 0.5), 0),
    1,
    viewMatrix,
  );

  mat4.multiply(projectionMatrix, viewMatrix, modelViewProjectionMatrix);

  return modelViewProjectionMatrix;
};

const cellShaderModule = device.createShaderModule({
  label: "Cell shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<u32>;
    @group(0) @binding(3) var<uniform> projectionMatrix: mat4x4f;

    struct VertexInput {
      @location(0) pos: vec4f,
      @builtin(instance_index) instance: u32,
    };
    
    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) cell: vec2f,
    };

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      let i = f32(input.instance);
      let cell = vec2f(i % grid.x, floor(i / grid.x));
      let state = f32(cellState[input.instance]);

      let cellOffset = (cell * 2.0) / grid - 1.0;
      let cubeSize = 1.0 / grid.x * 0.6;
      
      let position = vec4f(input.pos.xyz * cubeSize * state, 1.0);

      let modelMatrix = mat4x4f(
        vec4f(1, 0, 0, 0),
        vec4f(0, 1, 0, 0),
        vec4f(0, 0, 1, 0),
        vec4f(cellOffset, 0.0, 1)
      );

      var output: VertexOutput;
      output.pos = projectionMatrix * modelMatrix * position;
      output.cell = cell;
      return output;
    }

    struct FragInput {
      @location(0) cell: vec2f,
    };

    @fragment
    fn fragmentMain(input: FragInput) -> @location(0) vec4f {
      let c = input.cell / grid;
      return vec4f(c, 1-c.x, 1);
    }
  `,
});

const simulationShaderModule = device.createShaderModule({
  label: "Game of Life simulation shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

    fn cellIndex(cell: vec2u) -> u32 {
      return (cell.y % u32(grid.y)) * u32(grid.x) +
            (cell.x % u32(grid.x));
    }

    fn cellActive(x: u32, y: u32) -> u32 {
      return cellStateIn[cellIndex(vec2(x, y))];
    }

    @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
      let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                            cellActive(cell.x+1, cell.y) +
                            cellActive(cell.x+1, cell.y-1) +
                            cellActive(cell.x, cell.y-1) +
                            cellActive(cell.x-1, cell.y-1) +
                            cellActive(cell.x-1, cell.y) +
                            cellActive(cell.x-1, cell.y+1) +
                            cellActive(cell.x, cell.y+1);
      
      let i = cellIndex(cell.xy);
      switch activeNeighbors {
        case 2: {
          cellStateOut[i] = cellStateIn[i];
        }
        case 3: {
          cellStateOut[i] = 1;
        }
        default: {
          cellStateOut[i] = 0;
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
  computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

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
  const transformationMatrix = getTransformationMatrix();
  device.queue.writeBuffer(
    uniformMatrixBuffer,
    0,
    transformationMatrix.buffer,
    transformationMatrix.byteOffset,
    transformationMatrix.byteLength,
  );

  renderPass.setPipeline(cellPipeline);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.setBindGroup(0, bindGroups[step % 2]);

  renderPass.draw(verticesArray.length / 4, GRID_SIZE * GRID_SIZE);

  renderPass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(updateGrid);
};

requestAnimationFrame(updateGrid);
