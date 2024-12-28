export default class Renderer {
  public device: GPUDevice;
  public ctx: GPUCanvasContext;

  private constructor(device: GPUDevice, context: GPUCanvasContext) {
    this.device = device;
    this.ctx = context;
  }

  static async create(canvas: HTMLCanvasElement) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No appropriate GPUAdapter found.");

    const device = await adapter.requestDevice();
    if (!device) throw new Error("WebGPU not supported on this browser.");

    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("No canvas context found.");

    return new Renderer(device, context);
  }
}
