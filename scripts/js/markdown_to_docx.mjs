import fs from "node:fs/promises";
import path from "node:path";
import {
  BorderStyle,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  TextRun,
} from "docx";

const HEADING_MAP = {
  1: HeadingLevel.HEADING_1,
  2: HeadingLevel.HEADING_2,
  3: HeadingLevel.HEADING_3,
  4: HeadingLevel.HEADING_4,
  5: HeadingLevel.HEADING_5,
  6: HeadingLevel.HEADING_6,
};

function parseArgs(argv) {
  const args = { input: "", output: "" };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--input" || token === "-i") {
      args.input = argv[i + 1] || "";
      i += 1;
      continue;
    }
    if (token === "--output" || token === "-o") {
      args.output = argv[i + 1] || "";
      i += 1;
    }
  }
  return args;
}

function parseInlineRuns(text) {
  const chunks = text.split(/(`[^`]*`)/g).filter(Boolean);
  const runs = [];
  for (const chunk of chunks) {
    if (chunk.startsWith("`") && chunk.endsWith("`")) {
      runs.push(
        new TextRun({
          text: chunk.slice(1, -1),
          font: "Consolas",
          size: 20,
          color: "1F4E79",
        }),
      );
      continue;
    }
    runs.push(
      new TextRun({
        text: chunk,
        font: "宋体",
        size: 24,
      }),
    );
  }
  return runs.length > 0 ? runs : [new TextRun("")];
}

function parseMarkdownToParagraphs(markdownText) {
  const lines = markdownText.replace(/\r\n/g, "\n").split("\n");
  const paragraphs = [];

  for (const rawLine of lines) {
    const line = rawLine ?? "";
    const trimmed = line.trim();

    if (!trimmed) {
      paragraphs.push(
        new Paragraph({
          children: [new TextRun("")],
          spacing: { after: 160 },
        }),
      );
      continue;
    }

    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      paragraphs.push(
        new Paragraph({
          border: {
            bottom: {
              color: "D9D9D9",
              space: 1,
              size: 8,
              style: BorderStyle.SINGLE,
            },
          },
          spacing: { before: 120, after: 120 },
        }),
      );
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const title = headingMatch[2].trim();
      paragraphs.push(
        new Paragraph({
          heading: HEADING_MAP[level] ?? HeadingLevel.HEADING_6,
          children: parseInlineRuns(title),
          spacing: { before: level <= 2 ? 240 : 180, after: 120 },
        }),
      );
      continue;
    }

    const bulletMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
    if (bulletMatch) {
      const indentSpaces = bulletMatch[1].length;
      const level = Math.max(0, Math.min(8, Math.floor(indentSpaces / 2)));
      const content = bulletMatch[2].trim();
      paragraphs.push(
        new Paragraph({
          children: parseInlineRuns(content),
          bullet: { level },
          spacing: { after: 80 },
        }),
      );
      continue;
    }

    paragraphs.push(
      new Paragraph({
        children: parseInlineRuns(trimmed),
        spacing: { after: 120 },
      }),
    );
  }

  return paragraphs;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.input) {
    console.error(
      "Usage: node scripts/js/markdown_to_docx.mjs --input <input.md> [--output <output.docx>]",
    );
    process.exit(1);
  }

  const inputPath = path.resolve(args.input);
  const outputPath = path.resolve(
    args.output || inputPath.replace(/\.md$/i, ".docx"),
  );

  const markdown = await fs.readFile(inputPath, "utf-8");
  const content = parseMarkdownToParagraphs(markdown);

  const doc = new Document({
    sections: [
      {
        properties: {},
        children: content,
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  await fs.writeFile(outputPath, buffer);

  console.log(`DOCX generated: ${outputPath}`);
}

main().catch((err) => {
  console.error("Failed to generate DOCX:", err);
  process.exit(1);
});
