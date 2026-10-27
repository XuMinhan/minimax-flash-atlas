import { createModelRegistry } from "./registry";
import { minimaxM3 } from "./minimax-m3";
import { minimaxFlash } from "./minimax-m31-flash";

// Register verified model modules here. No disabled placeholder models.
export const modelRegistry = createModelRegistry([minimaxFlash,minimaxM3]);
