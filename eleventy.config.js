import markdownIt from "markdown-it";
import katex from "@vscode/markdown-it-katex";
import footnote_plugin from "markdown-it-footnote";

export default async function (eleventyConfig) {
  let md_options = {
    html: true,
    breaks: true,
    linkify: true,
    typographer: true,
  };

  const md = markdownIt(md_options)
    .use(katex.default, { throwOnError: true })
    .use(footnote_plugin);

  eleventyConfig.setLibrary("md", md);

  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/js");
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
