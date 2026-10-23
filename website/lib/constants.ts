export const LATEST_VERSION = "v1";
export const HEADER_HEIGHT = 30;
export const B2_BUCKT_NAME_ID_MAP = {
  "hubters-webai-public": "9bf1fe24e7edff5e924a091c",
};
export const ASSETS_ENDPOINT = "https://assets.axolsai.com";
export const API_ENDPOINT = "https://api.axolsai.com";

// Updated theme with direct color values for numbers and booleans
export const codeBlockTheme = {
  plain: {
    color: "var(--foreground)",
    backgroundColor: "transparent",
  },
  styles: [
    {
      types: ["comment", "prolog", "doctype", "cdata"],
      style: {
        color: "hsl(var(--muted-foreground))",
        // fontStyle: "italic",
      },
    },
    {
      types: ["namespace"],
      style: {
        color: "hsl(var(--muted-foreground))",
      },
    },
    {
      types: ["string", "attr-value","number","boolean"],
      style: {
        color: "#eb467c",
      },
    },
    {
      types: ["punctuation", "operator"],
      style: {
        color: "hsl(var(--foreground))",
      },
    },
    {
      types: ["entity", "url", "symbol", "inserted"],
      style: {
        color: "hsl(var(--warning))",
      },
    },


    {
      types: ["variable", "constant", "property", "regex"],
      style: {
        color: "hsl(var(--warning))",
      },
    },
    {
      types: ["atrule", "keyword", "attr-name", "selector"],
      style: {
        color: "hsl(var(--secondary))",
      },
    },
    {
      types: ["function", "deleted", "tag"],
      style: {
        color: "#f77b29",
      },
    },
    {
      types: ["function-variable"],
      style: {
        color: "hsl(var(--accent))",
      },
    },
    {
      types: ["tag", "selector", "keyword"],
      style: {
        color: "hsl(var(--secondary))",
      },
    },
  ],
};

