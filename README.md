# @axols/webai-js


[![npm version](https://img.shields.io/npm/v/@axols/webai-js.svg)](https://www.npmjs.com/package/@axols/webai-js)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/axols/webai-js/blob/main/LICENSE)

Run AI models directly in your users' browsers with zero server-side infrastructure.

**[📖 Documentation](https://www.webai-js.com/models/whisper-base/api-reference/v1/get-started/basic-usage)** | **[🎮 Playground](https://www.webai-js.com/models/whisper-base/playground)** | **[🤖 Models Hub](https://www.webai-js.com/models)** | **[👨🏻‍💻 Discord Community](https://discord.gg/RkpAKgZC)**


![test](https://assets.axolsai.com/web-assets/images/webai-js-banner.png)

## 🚀 Overview

Axols WebAI.js is an open-source library that enables client-side AI inference directly in the browser. Built on top of Transformers.js, WebGPU, and ONNX Runtime, it eliminates the need for server-side AI model hosting and inference infrastructure.

### Key Features

- 🏪 **ONNX Model Hub**: Access a curated pool of browser-optimized ONNX AI models
- 🌐 **Pure Client-Side**: Run AI models entirely in the browser
- 🔒 **Privacy-First**: Data never leaves the user's device
- 📦 **Zero Backend Costs**: No server infrastructure needed
- 🚀 **Easy Setup**: No headaches with packages - just one simple installation
- 🎯 **Standardized API**: Same interface across all models
- 🔄 **Streaming Support**: Real-time generation with streaming
- 🛠️ **Framework Compatible**: Works with React, Vue, Angular, Next.js, and more

## 📦 Installation

```bash
npm install @axols/webai-js
```

## 🎯 Quick Start

**All our Web AI models are standardized to use the same 3-step API interface:**

```javascript
import { WebAI } from '@axols/webai-js';

// Step 1: Create a WebAI instance
const webai = await WebAI.create({
  modelId: "llama-3.2-1b-instruct"
});

// Step 2: Initialize (downloads and loads the model)
await webai.init({
  mode: "auto", // Automatically selects best configuration
  onDownloadProgress: (progress) => {
    console.log(`Download progress: ${progress.progress}%`);
  }
});

// Step 3: Generate
const result = await webai.generate({
  userInput: {
     messages: [
      {
        role: "user",
        content: "What is the history of AI?"
      },
    ],   
  }
});

console.log(result);

// Step 4: Clean up when done
webai.terminate();
```

## 📖 Core Concepts

### Model Lifecycle

1. **Create**: Instantiate a WebAI object with a model ID
2. **Initialize**: Download (if needed) and load the model into memory
3. **Generate**: Run inference on user input
4. **Terminate**: Clean up resources when finished

### Auto Mode

Let WebAI automatically determine the best configuration based on device capabilities:

```javascript
await webai.init({
  mode: "auto",
  onDownloadProgress: (progress) => console.log(progress)
});
```

### Custom Priorities

Control fallback behavior with custom priority configurations:

```javascript
await webai.init({
  mode: "auto",
  priorities: [
    { mode: "webai", precision: "q4", device: "webgpu" },
    { mode: "webai", precision: "q8", device: "webgpu" },
    { mode: "webai", precision: "q4", device: "wasm" },
    { mode: "cloud", precision: "", device: "" }
  ]
});
```

## 🔄 Streaming Generation

For models that support streaming, provide real-time results:

```javascript
const generation = await webai.generateStream({
  userInput: "Tell me a story",
  onStream: (chunk) => {
    console.log(chunk); // Process each chunk as it arrives
  }
});
```

📚 **For detailed API reference and usage examples, see the [model-specific documentation](https://www.webai-js.com/models/whisper-base/api-reference/v1/get-started/basic-usage)**

## 💡 Best Practices

- ✅ Always wrap WebAI code in try/catch blocks
- ✅ Implement progress indicators during downloads
- ✅ Terminate instances when no longer needed
- ✅ Monitor device storage and memory usage
- ✅ Use streaming for better UX with long generations
- ✅ Test on target devices for performance validation

## 🌍 Browser Compatibility

WebAI.js works in all modern browsers that support:
- WebAssembly
- Web Workers
- WebGPU (recommended for best performance)

## 👥 Community

Join our growing community of developers building with WebAI.js!

- 💬 **[Discord](https://discord.gg/RkpAKgZC)** - Get help, share projects, understand AI trends, and help shape the future of Web AI
- 💡 **[Discussions](https://github.com/axols/webai-js/discussions)** - Share ideas and feature requests

## 🤝 Contributing

We welcome contributions! You're invited to add more AI models to our platform and contribute to the library. 

👨🏻‍💻 We are currently working on our Contributing Guide. In the meantime, feel free to [join our Discord](https://discord.gg/RkpAKgZC) to discuss how you can contribute!

🐛 For model-specific issues, bugs, or feature requests, please visit the [model issues page](https://www.webai-js.com/models/whisper-base/issues).

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE) file for details

## 🔗 Links

- [Homepage](https://www.webai-js.com)
- [Documentation](https://www.webai-js.com/models/whisper-base/api-reference/v1/get-started/basic-usage)
- [Model Hub](https://www.webai-js.com/models)
- [Playground](https://www.webai-js.com/models/whisper-base/playground)
- [Examples](https://github.com/axols/webai-examples)
- [Discord Community](https://discord.gg/RkpAKgZC)

## 🙏 Acknowledgments

Built with:
- [Transformers.js](https://github.com/xenova/transformers.js)
- [ONNX Runtime](https://onnxruntime.ai/)
- [WebGPU](https://www.w3.org/TR/webgpu/)

---

Made with ❤️ by Peng Zhang