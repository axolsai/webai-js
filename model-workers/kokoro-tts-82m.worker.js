import {
  StyleTextToSpeech2Model,
  AutoTokenizer,
  Tensor,
  RawAudio,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";
import { phonemize as espeakng } from "https://cdn.jsdelivr.net/npm/phonemizer@1.2.1/+esm";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "kokoro-tts-82m",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q4: {
      size: 305215966,
      price: 0.06,
      modelKeys: ["model_q4.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 4,
      },
    },
    q8: {
      size: 92361116,
      price: 0.03,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 3.8,
      },
    },
    uint8: {
      size: 177464632,
      price: 0.06,
      modelKeys: ["model_uint8.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 4,
      },
    },
    fp16: {
      size: 163234740,
      price: 0.06,
      modelKeys: ["model_fp16.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 4,
      },
    },
    fp32: {
      size: 325532232,
      price: 0.06,
      modelKeys: ["model.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 4,
      },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    voice: "af_bella",
    speed: 1.0,
    sample_rate: 24000,
    style_dim: 256,
  },
  DEFAULT_GENERATION_CONFIG: {
    return_full_audio: true,
  },
};

// ==========================================
// Global state
// ==========================================
let MODEL;
let modelState = {
  isInitializing: false,
  isInitialized: false,
  isGenerating: false,
};

// ==========================================
// Environment Setup
// ==========================================
function initializeEnvironment() {
  env.allowRemoteModels = true;
  env.remoteHost = "https://assets.axolsai.com";
  env.remotePathTemplate = "/models/{model}/model-repo/";
}

// ==========================================
// Utility Functions
// ==========================================
function checkModelSupports() {
  return {
    supportedModes: CONFIG.SUPPORTED_MODES,
    supportedPrecisions: Object.keys(CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP),
    supportedPrecisionsDevicesMap: CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP,
    doesSupportStreamGeneration: true,
    externalInterrupt: CONFIG.EXTERNAL_INTERRUPT,
  };
}

function mergeConfigs(defaults, userConfig) {
  return { ...defaults, ...userConfig };
}

export async function checkIsModelDownloaded(modelKeys, modelId) {
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();

    if (keys.length === 0) return false;

    // Build Set for O(1) lookups
    const keySet = new Set(keys.map((k) => k.url));

    return modelKeys.every((modelKey) =>
      Array.from(keySet).some(
        (url) => url.includes(modelKey) && url.includes(modelId),
      ),
    );
  } catch (error) {
    console.error("Error checking model download:", error);
    return false;
  }
}

// ==========================================
// Audio Utilities
// ==========================================
function concatenateWavFiles(wavBuffers, expectedSampleRate = 24000) {
  if (!wavBuffers || wavBuffers.length === 0) {
    throw new Error("No WAV buffers provided");
  }

  if (wavBuffers.length === 1) {
    return wavBuffers[0];
  }

  // Decode each WAV file to get clean audio data
  const audioSegments = [];
  let sampleRate = expectedSampleRate;
  let numChannels = 1;
  let bitsPerSample = 16;

  for (const wavBuffer of wavBuffers) {
    const view = new DataView(wavBuffer);

    // Parse WAV header
    const sr = view.getUint32(24, true);
    const nc = view.getUint16(22, true);
    const bps = view.getUint16(34, true);

    // Use header info from first file, but prefer expectedSampleRate
    if (audioSegments.length === 0) {
      sampleRate = expectedSampleRate || sr;
      numChannels = nc;
      bitsPerSample = bps;
    }

    // Get audio data size and position
    const dataSize = view.getUint32(40, true);
    const audioBytes = new Uint8Array(wavBuffer, 44, dataSize);

    // Convert to Float32Array for clean concatenation
    const samples = new Float32Array(dataSize / (bitsPerSample / 8));

    if (bitsPerSample === 16) {
      // Convert 16-bit PCM to float
      for (let i = 0; i < samples.length; i++) {
        const byte1 = audioBytes[i * 2];
        const byte2 = audioBytes[i * 2 + 1];
        const int16 = (byte2 << 8) | byte1;
        const signed = int16 > 32767 ? int16 - 65536 : int16;
        samples[i] = signed / 32768.0;
      }
    } else if (bitsPerSample === 32) {
      samples.set(
        new Float32Array(
          audioBytes.buffer,
          audioBytes.byteOffset,
          audioBytes.byteLength / 4,
        ),
      );
    }

    audioSegments.push(samples);
  }

  // Calculate total length
  const totalSamples = audioSegments.reduce((sum, seg) => sum + seg.length, 0);

  // Concatenate with small fade to avoid clicks
  const combinedAudio = new Float32Array(totalSamples);
  let offset = 0;

  for (let i = 0; i < audioSegments.length; i++) {
    const segment = audioSegments[i];

    // Apply small fade in/out at boundaries to reduce clicks
    if (i > 0) {
      const fadeLength = Math.min(100, segment.length);
      for (let j = 0; j < fadeLength; j++) {
        segment[j] *= j / fadeLength;
      }
    }

    if (i < audioSegments.length - 1) {
      const fadeLength = Math.min(100, segment.length);
      for (let j = 0; j < fadeLength; j++) {
        const idx = segment.length - fadeLength + j;
        segment[idx] *= (fadeLength - j) / fadeLength;
      }
    }

    combinedAudio.set(segment, offset);
    offset += segment.length;
  }

  // Convert back to 16-bit PCM WAV
  const dataSize = totalSamples * 2;
  const outputSize = 44 + dataSize;
  const outputBuffer = new ArrayBuffer(outputSize);
  const outputView = new DataView(outputBuffer);

  // Write WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      outputView.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  outputView.setUint32(4, outputSize - 8, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  outputView.setUint32(16, 16, true);
  outputView.setUint16(20, 1, true);
  outputView.setUint16(22, numChannels, true);
  outputView.setUint32(24, sampleRate, true);
  outputView.setUint32(28, sampleRate * numChannels * 2, true);
  outputView.setUint16(32, numChannels * 2, true);
  outputView.setUint16(34, 16, true);
  writeString(36, "data");
  outputView.setUint32(40, dataSize, true);

  // Convert float audio back to 16-bit PCM
  let byteOffset = 44;
  for (let i = 0; i < totalSamples; i++) {
    const sample = Math.max(-1, Math.min(1, combinedAudio[i]));
    const int16 = Math.round(sample * 32767);
    outputView.setInt16(byteOffset, int16, true);
    byteOffset += 2;
  }

  return outputBuffer;
}

// ==========================================
// Voice Utilities
// ==========================================
export const VOICES = Object.freeze({
  af_heart: {
    name: "Heart",
    language: "en-us",
    gender: "Female",
    traits: "❤️",
    targetQuality: "A",
    overallGrade: "A",
  },
  af_alloy: {
    name: "Alloy",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  af_aoede: {
    name: "Aoede",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  af_bella: {
    name: "Bella",
    language: "en-us",
    gender: "Female",
    traits: "🔥",
    targetQuality: "A",
    overallGrade: "A-",
  },
  af_jessica: {
    name: "Jessica",
    language: "en-us",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  af_kore: {
    name: "Kore",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  af_nicole: {
    name: "Nicole",
    language: "en-us",
    gender: "Female",
    traits: "🎧",
    targetQuality: "B",
    overallGrade: "B-",
  },
  af_nova: {
    name: "Nova",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  af_river: {
    name: "River",
    language: "en-us",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  af_sarah: {
    name: "Sarah",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  af_sky: {
    name: "Sky",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C-",
  },
  am_adam: {
    name: "Adam",
    language: "en-us",
    gender: "Male",
    targetQuality: "D",
    overallGrade: "F+",
  },
  am_echo: {
    name: "Echo",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_eric: {
    name: "Eric",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_fenrir: {
    name: "Fenrir",
    language: "en-us",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C+",
  },
  am_liam: {
    name: "Liam",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_michael: {
    name: "Michael",
    language: "en-us",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C+",
  },
  am_onyx: {
    name: "Onyx",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_puck: {
    name: "Puck",
    language: "en-us",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C+",
  },
  am_santa: {
    name: "Santa",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D-",
  },
  bf_emma: {
    name: "Emma",
    language: "en-gb",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "B-",
  },
  bf_isabella: {
    name: "Isabella",
    language: "en-gb",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  bm_george: {
    name: "George",
    language: "en-gb",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C",
  },
  bm_lewis: {
    name: "Lewis",
    language: "en-gb",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D+",
  },
  bf_alice: {
    name: "Alice",
    language: "en-gb",
    gender: "Female",
    traits: "🚺",
    targetQuality: "C",
    overallGrade: "D",
  },
  bf_lily: {
    name: "Lily",
    language: "en-gb",
    gender: "Female",
    traits: "🚺",
    targetQuality: "C",
    overallGrade: "D",
  },
  bm_daniel: {
    name: "Daniel",
    language: "en-gb",
    gender: "Male",
    traits: "🚹",
    targetQuality: "C",
    overallGrade: "D",
  },
  bm_fable: {
    name: "Fable",
    language: "en-gb",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  zf_xiaobei: {
    name: "XiaoBei",
    language: "en-cn",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  zf_xiaoyi: {
    name: "XiaoYi",
    language: "en-cn",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  zf_xiaoni: {
    name: "XiaoNi",
    language: "en-cn",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  zf_xiaoxiao: {
    name: "XiaoXiao",
    language: "en-cn",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  zm_yunjian: {
    name: "YunJian",
    language: "en-cn",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  zm_yunyang: {
    name: "YunYang",
    language: "en-cn",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  zm_yunxia: {
    name: "YunXia",
    language: "en-cn",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  zm_yunxi: {
    name: "YunXi",
    language: "en-cn",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  ef_dora: {
    name: "Dora",
    language: "ef",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  em_alex: {
    name: "Alex",
    language: "em",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  em_santa: {
    name: "Santa",
    language: "em",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  ff_siwis: {
    name: "Siwis",
    language: "ff",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  hf_alpha: {
    name: "Alpha",
    language: "hf",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  hf_beta: {
    name: "Beta",
    language: "hf",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  hm_omega: {
    name: "Omega",
    language: "hm",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  hm_psi: {
    name: "Psi",
    language: "hm",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  if_sara: {
    name: "Sara",
    language: "if",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  im_nicola: {
    name: "Nicola",
    language: "im",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  jf_alpha: {
    name: "Alpha",
    language: "jf",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  jf_gongitsune: {
    name: "Gongitsune",
    language: "jf",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  jf_nezumi: {
    name: "Nezumi",
    language: "jf",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  jf_tebukuro: {
    name: "Tebukuro",
    language: "jf",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  jm_kumo: {
    name: "Kumo",
    language: "jm",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  pf_dora: {
    name: "Dora",
    language: "pf",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "C",
  },
  pm_alex: {
    name: "Alex",
    language: "pm",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
  pm_santa: {
    name: "Santa",
    language: "pm",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },
});

const VOICE_DATA_URL = `https://assets.axolsai.com/models/${CONFIG.MODEL_ID}/model-repo/voices`;

async function getVoiceFile(id) {
  const url = `${VOICE_DATA_URL}/${id}.bin`;

  let cache;
  try {
    cache = await caches.open("kokoro-voices");
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
      return await cachedResponse.arrayBuffer();
    }
  } catch (e) {
    console.warn("Unable to open cache", e);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch voice file: ${response.status} ${response.statusText}`,
    );
  }

  const buffer = await response.arrayBuffer();

  if (cache) {
    try {
      await cache.put(
        url,
        new Response(buffer, {
          headers: response.headers,
        }),
      );
    } catch (e) {
      console.warn("Unable to cache file", e);
    }
  }

  return buffer;
}

const VOICE_CACHE = new Map();
async function getVoiceData(voice) {
  if (VOICE_CACHE.has(voice)) {
    return VOICE_CACHE.get(voice);
  }

  const buffer = new Float32Array(await getVoiceFile(voice));
  VOICE_CACHE.set(voice, buffer);
  return buffer;
}

// ==========================================
// Text Processing Utilities
// ==========================================
function split(text, regex) {
  const result = [];
  let prev = 0;
  for (const match of text.matchAll(regex)) {
    const fullMatch = match[0];
    if (prev < match.index) {
      result.push({ match: false, text: text.slice(prev, match.index) });
    }
    if (fullMatch.length > 0) {
      result.push({ match: true, text: fullMatch });
    }
    prev = match.index + fullMatch.length;
  }
  if (prev < text.length) {
    result.push({ match: false, text: text.slice(prev) });
  }
  return result;
}

function split_num(match) {
  if (match.includes(".")) {
    return match;
  } else if (match.includes(":")) {
    let [h, m] = match.split(":").map(Number);
    if (m === 0) {
      return `${h} o'clock`;
    } else if (m < 10) {
      return `${h} oh ${m}`;
    }
    return `${h} ${m}`;
  }
  let year = parseInt(match.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) {
    return match;
  }
  let left = match.slice(0, 2);
  let right = parseInt(match.slice(2, 4), 10);
  let suffix = match.endsWith("s") ? "s" : "";
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) {
      return `${left} hundred${suffix}`;
    } else if (right < 10) {
      return `${left} oh ${right}${suffix}`;
    }
  }
  return `${left} ${right}${suffix}`;
}

function flip_money(match) {
  const bill = match[0] === "$" ? "dollar" : "pound";
  if (isNaN(Number(match.slice(1)))) {
    return `${match.slice(1)} ${bill}s`;
  } else if (!match.includes(".")) {
    let suffix = match.slice(1) === "1" ? "" : "s";
    return `${match.slice(1)} ${bill}${suffix}`;
  }
  const [b, c] = match.slice(1).split(".");
  const d = parseInt(c.padEnd(2, "0"), 10);
  let coins =
    match[0] === "$"
      ? d === 1
        ? "cent"
        : "cents"
      : d === 1
        ? "penny"
        : "pence";
  return `${b} ${bill}${b === "1" ? "" : "s"} and ${d} ${coins}`;
}

function point_num(match) {
  let [a, b] = match.split(".");
  return `${a} point ${b.split("").join(" ")}`;
}

function normalize_text(text) {
  return (
    text
      // 1. Handle quotes and brackets
      .replace(/[‘’]/g, "'")
      .replace(/«/g, "“")
      .replace(/»/g, "”")
      .replace(/[“”]/g, '"')
      .replace(/\(/g, "«")
      .replace(/\)/g, "»")

      // 2. Replace uncommon punctuation marks
      .replace(/、/g, ", ")
      .replace(/。/g, ". ")
      .replace(/！/g, "! ")
      .replace(/，/g, ", ")
      .replace(/：/g, ": ")
      .replace(/；/g, "; ")
      .replace(/？/g, "? ")

      // 3. Whitespace normalization
      .replace(/[^\S \n]/g, " ")
      .replace(/  +/, " ")
      .replace(/(?<=\n) +(?=\n)/g, "")

      // 4. Abbreviations
      .replace(/\bD[Rr]\.(?= [A-Z])/g, "Doctor")
      .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, "Mister")
      .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, "Miss")
      .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, "Mrs")
      .replace(/\betc\.(?! [A-Z])/gi, "etc")

      // 5. Normalize casual words
      .replace(/\b(y)eah?\b/gi, "$1e'a")

      // 5. Handle numbers and currencies
      .replace(
        /\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g,
        split_num,
      )
      .replace(/(?<=\d),(?=\d)/g, "")
      .replace(
        /[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi,
        flip_money,
      )
      .replace(/\d*\.\d+/g, point_num)
      .replace(/(?<=\d)-(?=\d)/g, " to ")
      .replace(/(?<=\d)S/g, " S")

      // 6. Handle possessives
      .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
      .replace(/(?<=X')S\b/g, "s")

      // 7. Handle hyphenated words/letters
      .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, "-"))
      .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, "-")

      // 8. Strip leading and trailing whitespace
      .trim()
  );
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const PUNCTUATION = ';:,.!?¡¿—…"«»""(){}[]';
const PUNCTUATION_PATTERN = new RegExp(
  `(\\s*[${escapeRegExp(PUNCTUATION)}]+\\s*)+`,
  "g",
);

async function phonemize(text, language = "a", norm = true) {
  if (norm) {
    text = normalize_text(text);
  }

  const sections = split(text, PUNCTUATION_PATTERN);

  const lang = language === "a" ? "en-us" : "en";
  const ps = (
    await Promise.all(
      sections.map(async ({ match, text }) =>
        match ? text : (await espeakng(text, lang)).join(" "),
      ),
    )
  ).join("");

  let processed = ps
    .replace(/kəkˈoːɹoʊ/g, "kˈoʊkəɹoʊ")
    .replace(/kəkˈɔːɹəʊ/g, "kˈəʊkəɹəʊ")
    .replace(/ʲ/g, "j")
    .replace(/r/g, "ɹ")
    .replace(/x/g, "k")
    .replace(/ɬ/g, "l")
    .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, " ")
    .replace(/ z(?=[;:,.!?¡¿—…"«»"" ]|$)/g, "z");

  if (language === "a") {
    processed = processed.replace(/(?<=nˈaɪn)ti(?!ː)/g, "di");
  }
  return processed.trim();
}

// ==========================================
// Text Splitter Utilities
// ==========================================
function isSentenceTerminator(c, includeNewlines = true) {
  return ".!?…。？！".includes(c) || (includeNewlines && c === "\n");
}

function isTrailingChar(c) {
  return "\"')]}」』".includes(c);
}

function getTokenFromBuffer(buffer, start) {
  let end = start;
  while (end < buffer.length && !/\s/.test(buffer[end])) {
    ++end;
  }
  return buffer.substring(start, end);
}

const ABBREVIATIONS = new Set([
  "mr",
  "mrs",
  "ms",
  "dr",
  "prof",
  "sr",
  "jr",
  "sgt",
  "col",
  "gen",
  "rep",
  "sen",
  "gov",
  "lt",
  "maj",
  "capt",
  "st",
  "mt",
  "etc",
  "co",
  "inc",
  "ltd",
  "dept",
  "vs",
  "p",
  "pg",
  "jan",
  "feb",
  "mar",
  "apr",
  "jun",
  "jul",
  "aug",
  "sep",
  "sept",
  "oct",
  "nov",
  "dec",
  "sun",
  "mon",
  "tu",
  "tue",
  "tues",
  "wed",
  "th",
  "thu",
  "thur",
  "thurs",
  "fri",
  "sat",
]);

function isAbbreviation(token) {
  token = token.replace(/['']s$/i, "").replace(/\.+$/, "");
  return ABBREVIATIONS.has(token.toLowerCase());
}

const MATCHING = new Map([
  [")", "("],
  ["]", "["],
  ["}", "{"],
  ["》", "《"],
  ["〉", "〈"],
  ["›", "‹"],
  ["»", "«"],
  ["〉", "〈"],
  ["」", "「"],
  ["』", "『"],
  ["〕", "〔"],
  ["】", "【"],
]);
const OPENING = new Set(MATCHING.values());

function updateStack(c, stack, i, buffer) {
  if (c === '"' || c === "'") {
    if (
      c === "'" &&
      i > 0 &&
      i < buffer.length - 1 &&
      /[A-Za-z]/.test(buffer[i - 1]) &&
      /[A-Za-z]/.test(buffer[i + 1])
    ) {
      return;
    }
    if (stack.length && stack.at(-1) === c) {
      stack.pop();
    } else {
      stack.push(c);
    }
    return;
  }
  if (OPENING.has(c)) {
    stack.push(c);
    return;
  }
  const expectedOpening = MATCHING.get(c);
  if (expectedOpening && stack.length && stack.at(-1) === expectedOpening) {
    stack.pop();
  }
}

export class TextSplitterStream {
  constructor() {
    this._buffer = "";
    this._sentences = [];
    this._resolver = null;
    this._closed = false;
  }

  push(...texts) {
    for (const txt of texts) {
      this._buffer += txt;
      this._process();
    }
  }

  close() {
    if (this._closed) {
      throw new Error("Stream is already closed.");
    }
    this._closed = true;
    this.flush();
  }

  flush() {
    const remainder = this._buffer.trim();
    if (remainder.length > 0) {
      this._sentences.push(remainder);
    }
    this._buffer = "";
    this._resolve();
  }

  _resolve() {
    if (this._resolver) {
      this._resolver();
      this._resolver = null;
    }
  }

  _process() {
    let sentenceStart = 0;
    const buffer = this._buffer;
    const len = buffer.length;
    let i = 0;
    let stack = [];

    const scanBoundary = (idx) => {
      let end = idx;
      while (end + 1 < len && isSentenceTerminator(buffer[end + 1], false)) {
        ++end;
      }
      while (end + 1 < len && isTrailingChar(buffer[end + 1])) {
        ++end;
      }
      let nextNonSpace = end + 1;
      while (nextNonSpace < len && /\s/.test(buffer[nextNonSpace])) {
        ++nextNonSpace;
      }
      return { end, nextNonSpace };
    };

    while (i < len) {
      const c = buffer[i];
      updateStack(c, stack, i, buffer);

      if (stack.length === 0 && isSentenceTerminator(c)) {
        const currentSegment = buffer.slice(sentenceStart, i);
        if (/(^|\n)\d+$/.test(currentSegment)) {
          ++i;
          continue;
        }

        const { end: boundaryEnd, nextNonSpace } = scanBoundary(i);

        if (i === nextNonSpace - 1 && c !== "\n") {
          ++i;
          continue;
        }

        if (nextNonSpace === len) {
          break;
        }

        let tokenStart = i - 1;
        while (tokenStart >= 0 && /\S/.test(buffer[tokenStart])) {
          tokenStart--;
        }
        tokenStart = Math.max(sentenceStart, tokenStart + 1);
        const token = getTokenFromBuffer(buffer, tokenStart);
        if (!token) {
          ++i;
          continue;
        }

        if (
          (/https?[,:]\/\//.test(token) || token.includes("@")) &&
          !isSentenceTerminator(token.at(-1))
        ) {
          i = tokenStart + token.length;
          continue;
        }

        if (isAbbreviation(token)) {
          ++i;
          continue;
        }

        if (
          /^([A-Za-z]\.)+$/.test(token) &&
          nextNonSpace < len &&
          /[A-Z]/.test(buffer[nextNonSpace])
        ) {
          ++i;
          continue;
        }

        if (
          c === "." &&
          nextNonSpace < len &&
          /[a-z]/.test(buffer[nextNonSpace])
        ) {
          ++i;
          continue;
        }

        const sentence = buffer
          .substring(sentenceStart, boundaryEnd + 1)
          .trim();
        if (sentence === "..." || sentence === "…") {
          ++i;
          continue;
        }

        if (sentence) {
          this._sentences.push(sentence);
        }
        i = sentenceStart = boundaryEnd + 1;
        continue;
      }
      ++i;
    }

    this._buffer = buffer.substring(sentenceStart);

    if (this._sentences.length > 0) {
      this._resolve();
    }
  }

  async *[Symbol.asyncIterator]() {
    if (this._resolver) {
      throw new Error("Another iterator is already active.");
    }
    while (true) {
      if (this._sentences.length > 0) {
        yield this._sentences.shift();
      } else if (this._closed) {
        break;
      } else {
        await new Promise((resolve) => {
          this._resolver = resolve;
        });
      }
    }
  }

  [Symbol.iterator]() {
    this.flush();
    const iterator = this._sentences[Symbol.iterator]();
    this._sentences = [];
    return iterator;
  }

  get sentences() {
    return this._sentences;
  }
}

// ==========================================
// Model Class
// ==========================================
class WebAIModel {
  constructor() {
    this.model_id = CONFIG.MODEL_ID;
    this.model = null;
    this.tokenizer = null;
    this.precision = null;
    this.device = null;
  }

  setDefaults({ precision, device }) {
    this.precision = precision;
    this.device = device;
  }

  async downloadModel(precision) {
    if (!precision) {
      throw new Error("Precision must be set to download the model");
    }

    await AutoTokenizer.from_pretrained(this.model_id);
    await StyleTextToSpeech2Model.from_pretrained(this.model_id, {
      dtype: precision,
      progress_callback: (progress) => {
        self.postMessage({
          type: "downloadProgress",
          data: progress,
        });
      },
    });

    self.postMessage({
      type: "download",
      data: { status: "success" },
    });
  }

  async loadModel() {
    if (!this.precision || !this.device) {
      throw new Error(
        "Precision and device must be set via init before loading the model",
      );
    }

    const isDownloaded = await checkIsModelDownloaded(
      CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP[this.precision].modelKeys,
      this.model_id,
    );

    if (!isDownloaded) {
      throw new Error(
        "Model not downloaded. Call .init() method to download first.",
      );
    }

    this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);
    this.model = await StyleTextToSpeech2Model.from_pretrained(this.model_id, {
      device: this.device,
      dtype: this.precision,
      progress_callback: (progress) => {
        self.postMessage({
          type: "downloadProgress",
          data: progress,
        });
      },
    });
  }

  _validate_voice(voice) {
    if (!VOICES.hasOwnProperty(voice)) {
      console.error(`Voice "${voice}" not found. Available voices:`);
      console.table(VOICES);
      throw new Error(
        `Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`,
      );
    }
    return voice.at(0);
  }

  async generate_from_ids(
    input_ids,
    {
      voice = "af_heart",
      speed = 1,
      sample_rate = 24000,
      style_dim = 256,
    } = {},
  ) {
    const num_tokens = Math.min(Math.max(input_ids.dims.at(-1) - 2, 0), 509);
    const data = await getVoiceData(voice);
    const offset = num_tokens * style_dim;
    const voiceData = data.slice(offset, offset + style_dim);

    const inputs = {
      input_ids,
      style: new Tensor("float32", voiceData, [1, style_dim]),
      speed: new Tensor("float32", [speed], [1]),
    };

    const { waveform } = await this.model(inputs);
    return new RawAudio(waveform.data, sample_rate);
  }

  async generate(data) {
    const { userInput, modelConfig } = data;

    if (!this.model || !this.precision || !this.device) {
      throw new Error(
        "Model not initialized. You must call init first to set precision and device and load the model first.",
      );
    }

    if (modelState.isGenerating) {
      throw new Error(
        "A generation is already in progress. Please wait or interrupt the current generation.",
      );
    }

    modelState.isGenerating = true;

    try {
      if (!userInput.text || typeof userInput.text !== "string") {
        throw new Error(
          "Missing or invalid text input. Please provide a valid text string.",
        );
      }

      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );
      const language = this._validate_voice(finalModelConfig.voice);
      const phonemes = await phonemize(userInput.text, language);
      const { input_ids } = this.tokenizer(phonemes, { truncation: true });

      const output = await this.generate_from_ids(input_ids, {
        ...finalModelConfig,
      });

      const audioBuffer = output.toWav();
      const audioBlob = new Blob([audioBuffer], { type: "audio/wav" });
      const audioBlobUrl = URL.createObjectURL(audioBlob);

      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: { result: audioBlobUrl, result_text: userInput.text },
        },
      });
    } finally {
      modelState.isGenerating = false;
    }
  }

  async generateStream(data) {
    const { userInput, modelConfig, generateConfig } = data;

    if (!this.model || !this.precision || !this.device) {
      throw new Error(
        "Model not initialized. You must call init first to set precision and device and load the model first.",
      );
    }

    if (modelState.isGenerating) {
      throw new Error(
        "A generation is already in progress. Please wait or interrupt the current generation.",
      );
    }

    modelState.isGenerating = true;

    try {
      if (!userInput.text || typeof userInput.text !== "string") {
        throw new Error(
          "Missing or invalid text input. Please provide a valid text string.",
        );
      }

      const finalGenerateConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );
      const language = this._validate_voice(finalModelConfig.voice);

      let splitter;
      if (userInput.text instanceof TextSplitterStream) {
        splitter = userInput.text;
      } else if (typeof userInput.text === "string") {
        splitter = new TextSplitterStream();
        const chunks = userInput.split_pattern
          ? userInput.text
              .split(userInput.split_pattern)
              .map((chunk) => chunk.trim())
              .filter((chunk) => chunk.length > 0)
          : [userInput.text];
        splitter.push(...chunks);
        splitter.close();
      } else {
        throw new Error(
          "Invalid input type. Expected string or TextSplitterStream.",
        );
      }

      const wavSegments = [];

      for await (const sentence of splitter) {
        const phonemes = await phonemize(sentence, language);
        const { input_ids } = this.tokenizer(phonemes, { truncation: true });

        const output = await this.generate_from_ids(input_ids, {
          ...finalModelConfig,
        });

        const audioBuffer = output.toWav();
        wavSegments.push(audioBuffer);

        const audioBlob = new Blob([audioBuffer], { type: "audio/wav" });
        const audioBlobUrl = URL.createObjectURL(audioBlob);
        self.postMessage({
          type: "stream",
          data: {
            result: audioBlobUrl,
            result_segmented_text: sentence,
          },
        });
      }

      if (wavSegments.length > 0 && finalGenerateConfig.return_full_audio) {
        const concatenatedWav = concatenateWavFiles(
          wavSegments,
          finalModelConfig.sample_rate,
        );
        const fullAudioBlob = new Blob([concatenatedWav], {
          type: "audio/wav",
        });
        const fullAudioUrl = URL.createObjectURL(fullAudioBlob);

        self.postMessage({
          type: "generated",
          data: {
            status: "success",
            result: {
              result: fullAudioUrl,
              result_text: userInput.text,
            },
          },
        });
      } else {
        console.warn("No valid audio segments found");
        self.postMessage({
          type: "generated",
          data: {
            status: "success",
            result: { result: null },
          },
        });
      }
    } finally {
      modelState.isGenerating = false;
    }
  }
}

// ==========================================
// Message Handler
// ==========================================
async function handleMessage(event) {
  const { type, data } = event.data;
  console.log("Received message from main thread:", type);

  try {
    switch (type) {
      case "init":
        if (modelState.isInitializing) {
          throw new Error("Model is already initializing. Please wait.");
        }
        if (!data.precision || !data.device) {
          throw new Error(
            "Init message must contain both precision and device parameters",
          );
        }

        modelState.isInitializing = true;
        try {
          MODEL.setDefaults({
            precision: data.precision,
            device: data.device,
          });
          await MODEL.loadModel();
          modelState.isInitialized = true;
          self.postMessage({
            type: "init",
            data: {
              status: "success",
              precision: MODEL.precision,
              device: MODEL.device,
            },
          });
        } finally {
          modelState.isInitializing = false;
        }
        break;

      case "checkModelSupports":
        self.postMessage({
          type: "checkModelSupports",
          data: checkModelSupports(),
        });
        break;

      case "download":
        await MODEL.downloadModel(data.precision);
        break;

      case "generate":
        if (!modelState.isInitialized) {
          throw new Error(
            "Model not initialized. You must call init first to set precision and device.",
          );
        }

        await MODEL.generate({
          userInput: data.userInput || data,
          generateConfig: data.generateConfig || {},
          modelConfig: data.modelConfig || {},
        });
        break;

      case "generateStream":
        if (!modelState.isInitialized) {
          throw new Error(
            "Model not initialized. You must call init first to set precision and device.",
          );
        }

        await MODEL.generateStream(data);
        break;

      case "interrupt":
        // External interrupt - placeholder for future implementation
        break;

      case "clearMemory":
        MODEL.model = null;
        MODEL.tokenizer = null;
        modelState.isInitialized = false;
        modelState.isGenerating = false;
        self.postMessage({
          type: "clearMemory",
          data: { status: "success" },
        });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      data: {
        message: error.message || "Worker operation failed",
        context: type,
      },
    });
  }
}

// ==========================================
// Initialize Application
// ==========================================
function initializeApp() {
  // Global error handlers
  self.addEventListener("error", (event) => {
    self.postMessage({
      type: "error",
      data: {
        message: event.message,
        context: "global_error",
      },
    });
    event.preventDefault();
  });

  self.addEventListener("unhandledrejection", (event) => {
    self.postMessage({
      type: "error",
      data: {
        message: event.reason?.message || "Unhandled Promise Rejection",
        context: "unhandled_promise_rejection",
      },
    });
    event.preventDefault();
  });

  initializeEnvironment();
  MODEL = new WebAIModel();

  self.addEventListener("message", handleMessage);

  self.postMessage({
    type: "worker initialized",
    data: {
      success: true,
      message: "Web worker initialized successfully",
    },
  });
}

initializeApp();
