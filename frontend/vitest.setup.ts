import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Testing Library only auto-cleans when the test globals exist, so unmount
// explicitly between tests.
afterEach(() => {
  cleanup();
});
