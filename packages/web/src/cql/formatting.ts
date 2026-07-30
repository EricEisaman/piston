import type { ModuleSelectorToken, OpSelectorToken, ParameterSelectorToken } from "./types";

const COMBINATOR_LABELS: Record<string, string> = {
  child: ">",
  descendant: " ",
  "next-sibling": "+",
  "subsequent-sibling": "~",
};

/**
 * Format a module / op / parameter selector token for diagnostics and UI.
 */
export function formatSelectorToken(
  token: ModuleSelectorToken | OpSelectorToken | ParameterSelectorToken,
): string {
  if (token.source) {
    return token.source;
  }

  switch (token.type) {
    case "wildcard":
      return "*";
    case "name":
    case "type":
      return token.value;
    case "name-regex":
    case "type-regex":
    case "regex":
      return token.value instanceof RegExp ? token.value.source : String(token.value);
    case "combinator":
      return COMBINATOR_LABELS[token.kind] ?? token.kind;
    default:
      return String(token);
  }
}
