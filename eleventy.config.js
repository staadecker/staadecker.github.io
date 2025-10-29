import markdownIt from "markdown-it";
import katex from "@vscode/markdown-it-katex"

export default async function (eleventyConfig) {
  let options = {
    html: true,
    breaks: true,
    linkify: true,
    typographer: true,
  };

  eleventyConfig.setLibrary("md", markdownIt(options));
  eleventyConfig.amendLibrary("md", (mdLib) => mdLib.use(katex.default, { throwOnError: true }));

  eleventyConfig.addPassthroughCopy("src/css/fonts");
  eleventyConfig.addPassthroughCopy("src/img");
}

// This named export is optional
export const config = {
  dir: {
    input: "src",
    layouts: "_layouts",
    output: "dist",
  },
};
