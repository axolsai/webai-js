export type WebAIPrecision = "fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "bnb4" | "q4f16" ;
export type WebAIMode = "auto" | "webai" | "cloud";
export type WebAIDevice = "wasm" | "webgpu" 
export type ProgressType = {
    file: string;
    name: string;
    loaded: number;
    total: number;
    status: string;
    progress: number;
}

export type PriorityConfig = {
    mode: WebAIMode;
    precision: WebAIPrecision;
    device: WebAIDevice;
  };
  
export type WebAIPriorities = PriorityConfig[];


export type SupportedPrecisionsDevicesMapType = {
    [key in WebAIPrecision]: {
        supportedDevices: WebAIDevice[];
        precision: WebAIPrecision[];
        size: number;
        price: number;
        modelKeys: string[];
    }
}