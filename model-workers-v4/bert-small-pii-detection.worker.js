import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const ID = "bert-small-pii-detection";
const REPO = "onnx-community/bert-small-pii-detection-ONNX";
const REV = "6cb4e77c2b2c7f81e731b88cffa9b7a6fc675a4c";
const LABELS = ["AGE", "COORDINATE", "CREDIT_CARD", "DATE_TIME", "EMAIL_ADDRESS", "FINANCIAL", "IBAN_CODE", "IMEI", "IP_ADDRESS", "LOCATION", "MAC_ADDRESS", "NRP", "ORGANIZATION", "PASSWORD", "PERSON", "PHONE_NUMBER", "TITLE", "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_LICENSE_PLATE", "US_PASSPORT", "US_SSN"];
const PRECISIONS = {
  "fp32": {
    "weightsSize": 114181047,
    "size": 115128894,
    "dtype": "fp32",
    "modelKeys": [
      "model.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  },
  "bnb4": {
    "weightsSize": 71632769,
    "size": 72580616,
    "dtype": "bnb4",
    "modelKeys": [
      "model_bnb4.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  },
  "fp16": {
    "weightsSize": 57157730,
    "size": 58105577,
    "dtype": "fp16",
    "modelKeys": [
      "model_fp16.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  },
  "int8": {
    "weightsSize": 28748527,
    "size": 29696374,
    "dtype": "int8",
    "modelKeys": [
      "model_int8.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  },
  "q4": {
    "weightsSize": 71632769,
    "size": 72580616,
    "dtype": "q4",
    "modelKeys": [
      "model_q4.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  },
  "q4f16": {
    "weightsSize": 39033381,
    "size": 39981228,
    "dtype": "q4f16",
    "modelKeys": [
      "model_q4f16.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  },
  "q8": {
    "weightsSize": 28748527,
    "size": 29696374,
    "dtype": "q8",
    "modelKeys": [
      "model_quantized.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  },
  "uint8": {
    "weightsSize": 28748539,
    "size": 29696386,
    "dtype": "uint8",
    "modelKeys": [
      "model_uint8.onnx"
    ],
    "candidateDevices": [
      "wasm",
      "webgpu"
    ]
  }
};
const SUPPORTED = {};
const MANIFEST = {
  "contractVersion": "1.0",
  "model": {
    "id": "bert-small-pii-detection",
    "displayName": "BERT Small PII Detection",
    "provider": "Gravitee",
    "providerUrl": "https://huggingface.co/gravitee-io",
    "license": "Apache-2.0",
    "licenseUrl": "https://www.apache.org/licenses/LICENSE-2.0",
    "lastUpdated": "2026-05-21T09:54:18.000Z",
    "sourceRepository": "gravitee-io/bert-small-pii-detection",
    "sourceRevision": "f8c27a85c51c0168f07b9dcf00265bf0a4097939",
    "repository": "onnx-community/bert-small-pii-detection-ONNX",
    "artifactRevision": "6cb4e77c2b2c7f81e731b88cffa9b7a6fc675a4c",
    "artifactLastUpdated": "2026-04-10T02:43:30.000Z",
    "task": "token-classification",
    "description": "English PII token classification running locally in a browser. Returns typed entity spans with confidence scores and JavaScript character offsets for reviewed redaction workflows.",
    "intendedUses": [
      "Suggest PII spans for human-reviewed redaction of English support logs, documents, and API traces.",
      "Highlight personal names, contact details, and other sensitive entities before downstream processing."
    ],
    "limitations": [
      "English-focused; domain shifts, spelling errors, unusual formatting, and poor input quality can cause missed or incorrect entities.",
      "False negatives leave sensitive data exposed and false positives remove legitimate content; this is not a complete privacy or compliance guarantee.",
      "Only the first 512 tokenizer tokens are analyzed; split long documents explicitly and reconcile offsets.",
      "24 entity categories are available, but the fixture checks only PERSON, LOCATION, PHONE_NUMBER, and EMAIL_ADDRESS; other labels require domain-specific validation.",
      "Scores are not calibrated guarantees. Unknown or unalignable tokenizer output raises an error rather than returning unsafe redaction offsets."
    ]
  },
  "runtime": {
    "engine": {
      "name": "Transformers.js",
      "version": "4.2.0"
    },
    "precisions": {}
  },
  "input": {
    "description": "Non-whitespace English text; truncated at 512 tokenizer tokens including special tokens. Offsets refer to the original JavaScript string.",
    "alternatives": [
      {
        "name": "text",
        "fields": [
          {
            "name": "text",
            "type": "string",
            "required": true,
            "minLength": 1,
            "description": "Must contain non-whitespace text. Only the first 512 tokenizer tokens are analyzed."
          }
        ]
      }
    ]
  },
  "config": {
    "model": [],
    "generation": [
      {
        "name": "aggregation_strategy",
        "type": "string",
        "default": "simple",
        "enum": [
          "simple",
          "none"
        ],
        "description": "simple joins contiguous BIO entity tokens; none returns individual non-O token spans."
      }
    ]
  },
  "output": {
    "description": "Source-ordered entities. start is inclusive and end exclusive, measured in JavaScript UTF-16 code units.",
    "fields": [
      {
        "name": "entities",
        "type": "Array<{text:string,label:string,score:number,start:number,end:number}>",
        "description": "Labels: AGE, COORDINATE, CREDIT_CARD, DATE_TIME, EMAIL_ADDRESS, FINANCIAL, IBAN_CODE, IMEI, IP_ADDRESS, LOCATION, MAC_ADDRESS, NRP, ORGANIZATION, PASSWORD, PERSON, PHONE_NUMBER, TITLE, URL, US_BANK_NUMBER, US_DRIVER_LICENSE, US_ITIN, US_LICENSE_PLATE, US_PASSPORT, US_SSN"
      }
    ],
    "example": {
      "entities": [
        {
          "text": "Joshua Finch",
          "label": "PERSON",
          "score": 0.99,
          "start": 22,
          "end": 34
        }
      ]
    }
  },
  "operations": [
    "checkModelSupports",
    "init",
    "download",
    "generate",
    "clearMemory"
  ],
  "benchmark": {
  "report": "benchmarks/bert-small-pii-detection.json",
  "testedAt": "2026-10-23T14:27:01.466Z",
  "fixture": {
    "file": "test-app/public/test-prompts/bert-small-pii-detection.json",
    "url": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/bert-small-pii-detection.json",
    "size": 669,
    "sha1": "fc2f62bbf3218bc6f0622190e688310d5d9d9007",
    "source": "https://www.nasa.gov/news-release/nasa-spacex-invite-media-to-crew-8-mission-launch-to-space-station/",
    "license": "United States government work",
    "licenseUrl": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
    "preprocessing": "Verbatim English NASA newsroom contact block; BERT uncased tokenization, 512 token truncation, contiguous BIO aggregation.",
    "packagedFile": "benchmarks/bert-small-pii-detection-fixture.json"
  },
  "candidates": {
    "fp32": {
      "weightsSize": 114181047,
      "size": 115128894,
      "dtype": "fp32",
      "modelKeys": [
        "model.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    },
    "bnb4": {
      "weightsSize": 71632769,
      "size": 72580616,
      "dtype": "bnb4",
      "modelKeys": [
        "model_bnb4.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    },
    "fp16": {
      "weightsSize": 57157730,
      "size": 58105577,
      "dtype": "fp16",
      "modelKeys": [
        "model_fp16.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    },
    "int8": {
      "weightsSize": 28748527,
      "size": 29696374,
      "dtype": "int8",
      "modelKeys": [
        "model_int8.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    },
    "q4": {
      "weightsSize": 71632769,
      "size": 72580616,
      "dtype": "q4",
      "modelKeys": [
        "model_q4.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    },
    "q4f16": {
      "weightsSize": 39033381,
      "size": 39981228,
      "dtype": "q4f16",
      "modelKeys": [
        "model_q4f16.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    },
    "q8": {
      "weightsSize": 28748527,
      "size": 29696374,
      "dtype": "q8",
      "modelKeys": [
        "model_quantized.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    },
    "uint8": {
      "weightsSize": 28748539,
      "size": 29696386,
      "dtype": "uint8",
      "modelKeys": [
        "model_uint8.onnx"
      ],
      "candidateDevices": [
        "wasm",
        "webgpu"
      ]
    }
  },
  "qualityGate": "Exact labeled-span recall 1.0 at score >= 0.8 for all four checked contact entities, finite scores, ordered source-exact offsets, no empty or duplicate spans. This is a contact-span smoke gate, not general PII recall evidence.",
  "environment": "Isolated headless Google Chrome on macOS; Transformers.js 4.2.0",
  "invalidatedBy": [
    "artifact revision change",
    "runtime version change",
    "fixture or quality-gate change",
    "browser/backend behavior change"
  ],
  "results": [
    {
      "precision": "fp32",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": 6839.700000047684,
        "inferenceMs": 35.200000047683716,
        "totalMs": 6980.800000071526,
        "grade": "excellent",
        "textsPerSecond": 28.409090870606505
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": true,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9911609590053558,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7753128409385681,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9408915970060561,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "fp32",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": 14607.400000095367,
        "inferenceMs": 338.5,
        "totalMs": 15316.599999904633,
        "grade": "slow",
        "textsPerSecond": 2.9542097488921715
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": true,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9911610186100006,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7753126621246338,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.940891875161065,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "fp16",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": null,
        "inferenceMs": null,
        "totalMs": 4553.600000023842,
        "grade": "failed"
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "error": "Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: Type Error: Type (tensor(float16)) of output arg (/bert/encoder/layer.0/attention/self/Cast_output_0) of node (/bert/encoder/layer.0/attention/self/Cast) does not match expected type (tensor(float))."
    },
    {
      "precision": "fp16",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": null,
        "inferenceMs": null,
        "totalMs": 4507.899999976158,
        "grade": "failed"
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "error": "Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: Type Error: Type (tensor(float16)) of output arg (/bert/encoder/layer.0/attention/self/Cast_output_0) of node (/bert/encoder/layer.0/attention/self/Cast) does not match expected type (tensor(float))."
    },
    {
      "precision": "int8",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": 3856.0999999046326,
        "inferenceMs": 43.5,
        "totalMs": 3996.2999999523163,
        "grade": "excellent",
        "textsPerSecond": 22.988505747126435
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9876497685909271,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7417568564414978,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9573314719729953,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "int8",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": 3753.600000023842,
        "inferenceMs": 156.19999992847443,
        "totalMs": 4015.9000000953674,
        "grade": "slow",
        "textsPerSecond": 6.402048658501346
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9882926642894745,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7813637256622314,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9577475918663872,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "uint8",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": 4110.899999976158,
        "inferenceMs": 40.700000047683716,
        "totalMs": 4245.100000023842,
        "grade": "excellent",
        "textsPerSecond": 24.570024541238574
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": true,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9908840358257294,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.744717001914978,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9376120236184862,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "uint8",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": 3653.7999999523163,
        "inferenceMs": 84.79999995231628,
        "totalMs": 3839.399999976158,
        "grade": "good",
        "textsPerSecond": 11.792452836819669
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": true,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9903321266174316,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.8478608131408691,
            "pass": true
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9488459097014533,
            "pass": true
          }
        ],
        "recall": 0.75,
        "grade": "failed"
      }
    },
    {
      "precision": "q8",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": 3754.399999976158,
        "inferenceMs": 40.5,
        "totalMs": 3888.5,
        "grade": "excellent",
        "textsPerSecond": 24.691358024691358
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9876497685909271,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7417568564414978,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9573314719729953,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "q8",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": 3729.6999999284744,
        "inferenceMs": 91.10000002384186,
        "totalMs": 3962.899999976158,
        "grade": "good",
        "textsPerSecond": 10.976948405469694
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9882926642894745,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7813637256622314,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9577475918663872,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "q4",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": 6132,
        "inferenceMs": 52.700000047683716,
        "totalMs": 6282.899999976158,
        "grade": "good",
        "textsPerSecond": 18.975332051142043
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9882287979125977,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7136892676353455,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9377087288432651,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "q4",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": 5172.399999976158,
        "inferenceMs": 59.10000002384186,
        "totalMs": 5328.899999976158,
        "grade": "good",
        "textsPerSecond": 16.92047376643967
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.98822882771492,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7136889696121216,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9377084506882561,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "q4f16",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": null,
        "inferenceMs": null,
        "totalMs": 4716.599999904633,
        "grade": "failed"
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "error": "Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: Type Error: Type (tensor(float16)) of output arg (/bert/encoder/layer.0/attention/self/Cast_output_0) of node (/bert/encoder/layer.0/attention/self/Cast) does not match expected type (tensor(float))."
    },
    {
      "precision": "q4f16",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": null,
        "inferenceMs": null,
        "totalMs": 4291.100000023842,
        "grade": "failed"
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "error": "Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: Type Error: Type (tensor(float16)) of output arg (/bert/encoder/layer.0/attention/self/Cast_output_0) of node (/bert/encoder/layer.0/attention/self/Cast) does not match expected type (tensor(float))."
    },
    {
      "precision": "bnb4",
      "device": "wasm",
      "status": "fail",
      "runtime": {
        "loadMs": 5565.299999952316,
        "inferenceMs": 56.5,
        "totalMs": 5838.600000023842,
        "grade": "good",
        "textsPerSecond": 17.699115044247787
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.9882287979125977,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7136892676353455,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9377087288432651,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    },
    {
      "precision": "bnb4",
      "device": "webgpu",
      "status": "fail",
      "runtime": {
        "loadMs": 5437.100000023842,
        "inferenceMs": 57.699999928474426,
        "totalMs": 5603.100000023842,
        "grade": "good",
        "textsPerSecond": 17.33102255181302
      },
      "cleanup": {
        "modelDisposed": true,
        "cacheClearedBeforeCase": true,
        "cacheClearedAfterCase": true,
        "deletedEntries": 5
      },
      "quality": {
        "finite": true,
        "validOffsets": true,
        "nondegenerate": false,
        "expected": [
          {
            "text": "Joshua Finch",
            "label": "PERSON",
            "found": true,
            "score": 0.98822882771492,
            "pass": true
          },
          {
            "text": "Washington",
            "label": "LOCATION",
            "found": true,
            "score": 0.7136889696121216,
            "pass": false
          },
          {
            "text": "202-358-1100",
            "label": "PHONE_NUMBER",
            "found": false,
            "score": null,
            "pass": false
          },
          {
            "text": "joshua.a.finch@nasa.gov",
            "label": "EMAIL_ADDRESS",
            "found": true,
            "score": 0.9377084506882561,
            "pass": true
          }
        ],
        "recall": 0.5,
        "grade": "failed"
      }
    }
  ],
  "onboardingStatus": "blocked-quality-gate",
  "recommendedConfiguration": null,
  "blockers": [
    "No precision/device combination passes the checked-in contact-span quality gate. All successful inference cases omit the reference phone number. FP16 and Q4F16 fail ONNX session creation on both backends.",
    "A recommended configuration cannot be verified through the public WebAI API until a combination passes the quality gate."
  ],
  "runtimeProbe": {
    "passed": true,
    "moduleWorker": true,
    "engine": {
      "name": "Transformers.js",
      "version": "4.2.0"
    }
  }
}
};
MANIFEST.runtime.precisions = SUPPORTED;
MANIFEST.benchmark.candidates = PRECISIONS;
const state = { initializing: false, initialized: false, generating: false };
let classifier = null;
const send = (requestId, type, data) =>
  self.postMessage({ requestId, type, data });
const capabilities = {
  supportedModes: ["webai"],
  supportedPrecisions: Object.keys(SUPPORTED),
  supportedPrecisionsDevicesMap: SUPPORTED,
  doesSupportStreamGeneration: false,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: "4.2.0",
  cacheModelId: REPO,
};

function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate)
    env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}
function validateInit(data, supported = SUPPORTED) {
  const precision = data?.precision,
    device = data?.device,
    entry = supported[precision];
  if (!entry) throw Error(`Unsupported precision: ${precision}`);
  const devices = entry.supportedDevices ?? entry.candidateDevices;
  if (!devices.includes(device))
    throw Error(
      `Precision ${precision} is only supported on: ${devices.join(", ")}`,
    );
  if (device === "webgpu" && !self.navigator?.gpu)
    throw Error("WebGPU is unavailable");
  return { precision, device };
}
function options(data) {
  for (const [name, value] of [["modelConfig", data?.modelConfig], ["generateConfig", data?.generateConfig]]) {
    if (value !== undefined && (!value || typeof value !== "object" || Array.isArray(value))) throw Error(`${name} must be an object`);
    for (const key of Object.keys(value ?? {})) if (name === "modelConfig" || key !== "aggregation_strategy") throw Error(`Unsupported ${name} option: ${key}`);
  }
  const strategy = data?.generateConfig?.aggregation_strategy ?? MANIFEST.config.generation[0].default;
  if (!MANIFEST.config.generation[0].enum.includes(strategy)) throw Error("aggregation_strategy must be simple or none");
  return strategy;
}
async function clear() {
  if (classifier?.dispose) await classifier.dispose();
  classifier = null;
  state.initialized = false;
}
async function load(data, requestId, supported = SUPPORTED) {
  const { precision, device } = validateInit(data, supported);
  await clear();
  classifier = await pipeline("token-classification", REPO, {
    revision: REV,
    dtype: precision,
    device,
    progress_callback: (value) => send(requestId, "downloadProgress", value),
  });
}
// Include O tokens during alignment so repeated text cannot attach to an earlier occurrence.
async function generate(data) {
  const text = data?.userInput?.text;
  if (typeof text !== "string" || !text.trim()) throw Error("userInput.text must be a non-empty string");
  const strategy = options(data);
  const raw = await classifier(text, { ignore_labels: [], aggregation_strategy: "none" });
  const normalize = value => value.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
  let normalized = "";
  const starts = [], ends = [];
  for (let index = 0; index < text.length;) {
    const point = String.fromCodePoint(text.codePointAt(index));
    for (const char of normalize(point)) { normalized += char; starts.push(index); ends.push(index + point.length); }
    index += point.length;
  }
  let cursor = 0, open = null;
  const entities = [];
  for (const item of raw) {
    const needle = normalize(String(item.word).replace(/^##/, ""));
    const position = normalized.indexOf(needle, cursor);
    if (!needle || position < 0 || /\S/u.test(normalized.slice(cursor, position))) throw Error(`Cannot safely align token: ${item.word}`);
    const start = starts[position], end = ends[position + needle.length - 1];
    cursor = position + needle.length;
    const label = item.entity.replace(/^[BI]-/, ""), score = Number(item.score);
    if (!Number.isFinite(score) || score < 0 || score > 1) throw Error("Invalid entity score");
    if (label === "O") { open = null; continue; }
    if (!LABELS.includes(label)) throw Error(`Invalid entity label: ${label}`);
    if (strategy === "simple" && open && open.label === label && !item.entity.startsWith("B-") && item.index === open.lastIndex + 1) {
      open.end = end; open.text = text.slice(open.start, end); open.sum += score; open.count++; open.score = open.sum / open.count; open.lastIndex = item.index;
    } else {
      open = {text:text.slice(start,end), label, score, start, end, sum:score, count:1, lastIndex:item.index}; entities.push(open);
    }
  }
  return {entities:entities.map(({text,label,score,start,end})=>({text,label,score,start,end}))};
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") {
      configure(data?.workerConfig);
      send(requestId, type, { ...capabilities, manifest: MANIFEST });
      return;
    }
    if (type === "init" || type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      state.initialized = false;
      try {
        const precision = data?.precision,
          device =
            type === "download"
              ? SUPPORTED[precision]?.supportedDevices[0]
              : data?.device;
        await load({ ...data, precision, device }, requestId);
        if (type === "download") await clear();
        else state.initialized = true;
        send(requestId, type, { status: "success" });
      } finally {
        state.initializing = false;
      }
      return;
    }
    if (type === "generate") {
      if (state.initializing || !state.initialized || !classifier)
        throw Error("Call init before generate");
      if (state.generating) throw Error("Generation already in progress");
      state.generating = true;
      try {
        send(requestId, "generated", {
          status: "success",
          result: await generate(data),
        });
      } finally {
        state.generating = false;
      }
      return;
    }
    if (type === "generateStream")
      throw Error("Stream generation is not supported");
    if (type === "clearMemory") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      try {
        await clear();
        send(requestId, type, { status: "success" });
      } finally {
        state.initializing = false;
      }
      return;
    }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) {
    send(requestId, "error", {
      message: error instanceof Error ? error.message : String(error),
      context: type,
    });
  }
});
self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: "v4", modelId: ID },
});
