import type {
  Plugin,
  ContentConverter,
  ConversionInput,
  ConversionResult,
  ConverterStageDescriptor,
  MappingData,
} from '@yandu/types';
import { spawn } from 'child_process';
import { writeFile, mkdir, readFile } from 'fs/promises';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { createServer } from 'net';

// ─── MinerU API types ───

interface MinerUApiResponse {
  task_id?: string;
  status?: string;
  backend: string;
  version: string;
  results: Record<string, MinerUFileResult>;
}

interface MinerUContentItem {
  type: string;
  img_path?: string;
  image_caption?: string[];
  image_footnote?: string[];
  table_caption?: string[];
  table_footnote?: string[];
  table_body?: string;
  list_items?: string[];
  code_body?: string;
  sub_type?: string;
  bbox?: number[];
  page_idx?: number;
  text?: string;
  text_level?: number;
  text_format?: string;
}

interface MinerUFileResult {
  md_content?: string;
  content_list?: MinerUContentItem[];
  images?: Record<string, string>;
  middle_json?: Record<string, unknown>;
  model_output?: unknown;
}

interface MinerUResult {
  markdown: string;
  images: Array<{ filename: string; page: number; bbox: [number, number, number, number] }>;
  decodedImages: Record<string, string>;
  pages: number;
}

// ─── TQDM stderr parser ───

const TQDM_REGEX = /^(.+?):\s+(\d+)%\|.*?\|\s+(\d+)\/(\d+)/;

function parseTqdmLine(line: string): { stageName: string; percent: number; current: number; total: number } | null {
  const match = line.match(TQDM_REGEX);
  if (!match) return null;
  return {
    stageName: match[1].trim(),
    percent: parseInt(match[2], 10),
    current: parseInt(match[3], 10),
    total: parseInt(match[4], 10),
  };
}

const STAGE_NAME_MAP: Record<string, string> = {
  'Layout Predict': 'layout_predict',
  'MFR Predict': 'mfr_predict',
  'Table-ocr det': 'table_ocr',
  'Table-ocr rec ch': 'table_ocr',
  'Table-ocr rec en': 'table_ocr',
  'Table-wired Predict': 'table_ocr',
  'OCR-det ch': 'ocr',
  'OCR-det en': 'ocr',
  'OCR-rec Predict': 'ocr',
  'Processing pages': 'ocr',
};

function mapStageNameToId(stageName: string): string | null {
  return STAGE_NAME_MAP[stageName] ?? null;
}

// ─── Utilities ───

function findFreePort(startPort = 18000): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.on('error', reject);
    server.listen(startPort, '127.0.0.1', () => {
      const port = (server.address() as { port: number }).port;
      server.close(() => resolve(port));
    });
  });
}

function detectGpu(): 'cuda' | 'mps' | 'none' {
  try {
    const { execSync } = require('child_process');
    if (process.platform !== 'darwin') {
      execSync('nvidia-smi', { stdio: 'ignore' });
      return 'cuda';
    }
  } catch {
    // no CUDA
  }
  if (process.platform === 'darwin') {
    try {
      const { execSync } = require('child_process');
      const out = execSync('system_profiler SPDisplaysDataType', { encoding: 'utf-8' });
      if (out.includes('Apple M')) return 'mps';
    } catch {
      // ignore
    }
  }
  return 'none';
}

function gpuTypeToMinerUDeviceMode(gpu: string): string {
  if (gpu === 'cuda') return 'cuda';
  if (gpu === 'mps') return 'mps';
  return 'cpu';
}

function getMinerUPath(): string {
  return 'mineru-api';
}

function ensureModelscopePatch(): string {
  const runtimeDir = path.join(process.env.APPDATA || process.env.HOME || '.', '.yandu', 'runtime');
  const patchDir = path.join(runtimeDir, 'mineru-modelscope-patch');
  const patchFile = path.join(patchDir, 'modelscope.py');
  if (!existsSync(patchFile)) {
    mkdirSync(patchDir, { recursive: true });
    writeFileSync(
      patchFile,
      '# Auto-generated to prevent modelscope import hang\n' +
      'def snapshot_download(*args, **kwargs):\n' +
      '    raise RuntimeError("modelscope disabled; set MINERU_MODEL_SOURCE=huggingface")\n',
      'utf-8',
    );
  }
  return patchDir;
}

// ─── MinerU HTTP Service ───

class MinerUService {
  private managed: ReturnType<typeof spawn> | null = null;
  private port = 0;
  private baseUrl = '';
  private stopping = false;
  private startPromise: Promise<void> | null = null;
  private onTqdmProgress: ((tqdm: ReturnType<typeof parseTqdmLine>) => void) | null = null;

  setTqdmProgressCallback(cb: typeof this.onTqdmProgress): void {
    this.onTqdmProgress = cb;
  }

  async start(gpuAccel = 'off'): Promise<void> {
    if (this.managed) return;
    if (this.startPromise) return this.startPromise;

    this.startPromise = this.doStart(gpuAccel);
    try {
      await this.startPromise;
    } finally {
      this.startPromise = null;
    }
  }

  private async doStart(gpuAccel: string): Promise<void> {
    if (this.managed) return;
    this.port = await findFreePort(18000);
    this.baseUrl = `http://127.0.0.1:${this.port}`;

    const deviceMode = gpuAccel === 'off'
      ? 'cpu'
      : gpuAccel === 'auto'
        ? gpuTypeToMinerUDeviceMode(detectGpu())
        : gpuTypeToMinerUDeviceMode(gpuAccel);

    if (gpuAccel !== 'off' && deviceMode === 'cpu') {
      console.warn(`[MinerU] GPU acceleration requested (${gpuAccel}) but no supported GPU detected, falling back to CPU`);
    }

    const modelscopePatchDir = ensureModelscopePatch();
    const pythonPath = process.env.PYTHONPATH
      ? `${modelscopePatchDir}${path.delimiter}${process.env.PYTHONPATH}`
      : modelscopePatchDir;

    const mineruPath = getMinerUPath();

    const child = spawn(mineruPath, [
      '--host', '127.0.0.1',
      '--port', String(this.port),
      '--enable-vlm-preload', 'true',
    ], {
      env: {
        ...process.env,
        MINERU_DEVICE_MODE: deviceMode,
        MINERU_MODEL_SOURCE: 'huggingface',
        PYTHONPATH: pythonPath,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    child.stdout?.on('data', (chunk: Buffer) => {
      const lines = chunk.toString().split('\n');
      for (const line of lines) {
        if (line.trim()) console.log(`[MinerU] ${line.trim()}`);
      }
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      const lines = chunk.toString().split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        const tqdm = parseTqdmLine(trimmed);
        if (tqdm && this.onTqdmProgress) {
          this.onTqdmProgress(tqdm);
        }
        console.warn(`[MinerU] ${trimmed}`);
      }
    });

    child.on('exit', (code, signal) => {
      if (!this.stopping) {
        console.warn(`[MinerU] process exited unexpectedly (code=${code}, signal=${signal})`);
      }
      this.managed = null;
    });

    // Wait for spawn
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('MinerU spawn timeout')), 5000);
      child.on('spawn', () => {
        clearTimeout(timer);
        resolve();
      });
      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
    });

    // Wait for health endpoint
    const maxWaitMs = 600_000;
    const pollInterval = 500;
    const deadline = Date.now() + maxWaitMs;
    while (Date.now() < deadline) {
      try {
        const response = await fetch(`${this.baseUrl}/health`);
        if (response.ok) {
          this.managed = child;
          console.log(`[MinerU] HTTP API ready at ${this.baseUrl}`);
          return;
        }
      } catch {
        // not ready yet
      }
      await new Promise((r) => setTimeout(r, pollInterval));
    }
    child.kill();
    throw new Error('MinerU API failed to start within timeout');
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  async stop(timeoutMs = 5000): Promise<void> {
    const child = this.managed;
    if (!child || child.killed) {
      this.managed = null;
      return;
    }
    this.stopping = true;
    child.kill('SIGTERM');
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        resolve();
      }, timeoutMs);
      child.on('exit', () => {
        clearTimeout(timer);
        resolve();
      });
    });
    this.managed = null;
    this.stopping = false;
  }
}

const mineruService = new MinerUService();

// ─── PDFConverter ───

class PDFConverter implements ContentConverter {
  id = 'converter.pdf';
  inputFormats = ['application/pdf'];

  settingsSchema = {
    type: 'object',
    properties: {
      extractImages: { type: 'boolean', default: true },
      keepOriginal: { type: 'boolean', default: true },
      gpuAcceleration: { type: 'string', default: 'off' },
      serverUrl: { type: 'string', default: '' },
    },
  };

  getStages(): ConverterStageDescriptor[] {
    return [
      { id: 'prepare', label: 'Prepare File', weight: 0.05 },
      { id: 'start_server', label: 'Start Server', weight: 0.05 },
      { id: 'layout_predict', label: 'Layout Analysis', weight: 0.20 },
      { id: 'mfr_predict', label: 'Formula Recognition', weight: 0.15 },
      { id: 'table_ocr', label: 'Table OCR', weight: 0.15 },
      { id: 'ocr', label: 'Text OCR', weight: 0.20 },
      { id: 'save', label: 'Save Result', weight: 0.20 },
    ];
  }

  async convert(input: ConversionInput): Promise<ConversionResult> {
    const { source, outputDir, onProgress, settings = {} } = input;
    const s = settings as {
      extractImages?: boolean;
      keepOriginal?: boolean;
      gpuAcceleration?: string;
      serverUrl?: string;
    };

    const stages = this.getStages();
    const stageIdx = (id: string) => stages.findIndex((st) => st.id === id);

    const report = (stageId: string, stageProgress: number, detail?: string) => {
      const idx = stageIdx(stageId);
      let completedWeight = 0;
      for (let i = 0; i < idx; i++) completedWeight += stages[i].weight;
      const activeWeight = stages[idx]?.weight ?? 0;
      const overall = Math.round((completedWeight + activeWeight * (stageProgress / 100)) * 100);
      onProgress?.({ stageId, stageProgress, detail, overallProgress: overall });
    };

    // Prepare
    report('prepare', 0);
    await mkdir(outputDir, { recursive: true });

    let pdfPath: string;
    if (typeof source === 'string') {
      pdfPath = source;
    } else {
      const tempPath = path.join(outputDir, 'temp_input.pdf');
      const buffer = Buffer.from(await source.arrayBuffer());
      await writeFile(tempPath, buffer);
      pdfPath = tempPath;
    }
    report('prepare', 100);

    // Start server
    report('start_server', 0);
    await mineruService.start(s.gpuAcceleration ?? 'off');

    const hasCuda = detectGpu() === 'cuda';
    const userBackend = (s.serverUrl ? 'pipeline' : undefined) ?? 'auto';
    const backend = userBackend === 'pipeline'
      ? 'pipeline'
      : userBackend === 'auto' && !hasCuda
        ? 'pipeline'
        : 'hybrid-auto-engine';
    report('start_server', 100);

    // Parse with tqdm callback
    mineruService.setTqdmProgressCallback((tqdm) => {
      if (!tqdm) return;
      const mappedStageId = mapStageNameToId(tqdm.stageName);
      if (mappedStageId) {
        report(mappedStageId, tqdm.percent, `${tqdm.stageName}: ${tqdm.current}/${tqdm.total}`);
      }
    });

    let result: MinerUResult;
    try {
      result = await this.callMinerUAPI(pdfPath, {
        outputDir,
        extractImages: s.extractImages ?? true,
        backend,
        serverUrl: s.serverUrl,
      });

      if (!result.markdown.trim() && backend !== 'pipeline') {
        console.warn(`[PDFConverter] ${backend} produced empty markdown, falling back to pipeline`);
        result = await this.callMinerUAPI(pdfPath, {
          outputDir,
          extractImages: s.extractImages ?? true,
          backend: 'pipeline',
          serverUrl: s.serverUrl,
        });
      }
    } catch (vlmErr) {
      if (backend !== 'pipeline') {
        const errMsg = vlmErr instanceof Error ? vlmErr.message : String(vlmErr);
        console.warn(`[PDFConverter] ${backend} failed (${errMsg}), falling back to pipeline`);
        result = await this.callMinerUAPI(pdfPath, {
          outputDir,
          extractImages: s.extractImages ?? true,
          backend: 'pipeline',
          serverUrl: s.serverUrl,
        });
      } else {
        throw vlmErr;
      }
    } finally {
      mineruService.setTqdmProgressCallback(null);
    }

    report('ocr', 100);

    // Save
    report('save', 0);
    const assetsDir = path.join(outputDir, 'assets');
    await mkdir(assetsDir, { recursive: true });

    let finalMarkdown = result.markdown;
    for (const [filename, dataUrl] of Object.entries(result.decodedImages)) {
      const match = dataUrl.match(/^data:image\/\w+;base64,(.+)$/);
      if (!match) continue;
      const buf = Buffer.from(match[1], 'base64');
      await writeFile(path.join(assetsDir, filename), buf);
    }
    if (Object.keys(result.decodedImages).length > 0) {
      finalMarkdown = finalMarkdown.replace(/\]\(images\//g, '](assets/');
    }

    const mapping = this.generateMapping(result, pdfPath);
    const markdownPath = path.join(outputDir, 'output.md');
    await writeFile(markdownPath, finalMarkdown);

    const mappingPath = path.join(outputDir, 'paper.mapping.yaml');
    await writeFile(mappingPath, this.stringifyYaml(mapping));

    if (typeof source !== 'string' && pdfPath.includes('temp_input')) {
      const { unlink } = await import('fs/promises');
      await unlink(pdfPath).catch(() => {});
    }

    onProgress?.({ stageId: 'save', stageProgress: 100, overallProgress: 100 });

    return {
      markdown: finalMarkdown,
      markdownPath,
      mappingPath,
      assets: result.images.map((img, i) => ({
        id: `fig${i + 1}`,
        path: `assets/${img.filename}`,
        type: 'image' as const,
      })),
    };
  }

  private async callMinerUAPI(
    pdfPath: string,
    options: {
      outputDir: string;
      extractImages: boolean;
      backend: string;
      serverUrl?: string;
    }
  ): Promise<MinerUResult> {
    const fileBuffer = await readFile(pdfPath);
    const fileName = path.basename(pdfPath);
    const form = new FormData();
    form.append('files', new File([fileBuffer], fileName, { type: 'application/pdf' }));
    form.append('backend', options.backend);
    form.append('parse_method', 'auto');
    form.append('formula_enable', 'true');
    form.append('table_enable', 'true');
    form.append('return_md', 'true');
    form.append('return_content_list', String(options.extractImages));
    form.append('return_images', String(options.extractImages));

    if (options.serverUrl) {
      form.append('server_url', options.serverUrl);
    }

    const baseUrl = mineruService.getBaseUrl();
    console.log(`[PDFConverter] Calling MinerU API at ${baseUrl}/file_parse with backend=${options.backend}`);
    const response = await fetch(`${baseUrl}/file_parse`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(600_000),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => response.statusText);
      throw new Error(`MinerU API error ${response.status}: ${errorText}`);
    }

    const apiResponse = await response.json() as MinerUApiResponse;
    const fileResults = Object.values(apiResponse.results || {});
    if (fileResults.length === 0) {
      const legacy = apiResponse as unknown as { markdown?: string };
      if (legacy.markdown) {
        return { markdown: legacy.markdown, images: [], decodedImages: {}, pages: 0 };
      }
      throw new Error('MinerU API returned empty result');
    }

    const fileResult = fileResults[0];
    let markdown = fileResult.md_content || '';

    const images: MinerUResult['images'] = [];
    let rawContentList: unknown = fileResult.content_list;

    if (typeof rawContentList === 'string') {
      try {
        rawContentList = JSON.parse(rawContentList);
      } catch {
        // ignore parse error
      }
    }

    const rawArray: unknown[] = Array.isArray(rawContentList) ? rawContentList : [];
    const contentList = rawArray.filter(
      (item): item is MinerUContentItem =>
        item != null && typeof item === 'object' && typeof (item as { type?: unknown }).type === 'string'
    );

    // Rebuild markdown from content_list if md_content is empty
    let effectiveMarkdown = markdown;
    if (!effectiveMarkdown.trim() && contentList.length > 0) {
      const parts: string[] = [];
      for (const item of contentList) {
        const text = item.text ?? '';
        const t = item.type ?? '';
        if (t === 'text' || t === 'header' || t === 'footer' || t === 'page_number' || t === 'ref_text' || t === 'aside_text') {
          const level = item.text_level ?? 0;
          if (text.trim()) parts.push(level > 0 ? `${'#'.repeat(level)} ${text}` : text);
        } else if (t === 'title' || t === 'doc_title' || t === 'paragraph_title') {
          const level = item.text_level ?? 1;
          if (text.trim()) parts.push(`${'#'.repeat(level)} ${text}`);
        } else if (t === 'list') {
          for (const li of (item.list_items ?? [])) {
            if (li.trim()) parts.push(`- ${li}`);
          }
        } else if (t === 'equation' || t === 'interline_equation') {
          if (text.trim()) parts.push(`$$\n${text}\n$$`);
          else if (item.img_path) parts.push(`![](${item.img_path})`);
        } else if (t === 'image') {
          if (item.img_path) {
            parts.push(`![](${item.img_path})`);
            for (const cap of (item.image_caption ?? [])) {
              if (cap.trim()) parts.push(cap);
            }
          }
        } else if (t === 'table') {
          const body = item.table_body ?? '';
          if (body) parts.push(body);
          else if (item.img_path) parts.push(`![](${item.img_path})`);
        } else if (t === 'code') {
          const body = item.code_body ?? '';
          if (body) parts.push(body);
        } else if (t === 'chart') {
          if (item.img_path) parts.push(`![](${item.img_path})`);
        }
      }
      if (parts.length > 0) {
        effectiveMarkdown = parts.join('\n\n');
      }
    }

    for (const item of contentList) {
      if (item.type === 'image' && item.img_path) {
        images.push({
          filename: path.basename(item.img_path),
          page: item.page_idx ?? 0,
          bbox: (item.bbox?.slice(0, 4) as [number, number, number, number]) ?? [0, 0, 0, 0],
        });
      }
    }

    const maxPage = contentList.length > 0
      ? Math.max(...contentList.map((item) => item.page_idx ?? 0), 0) + 1
      : 0;

    return { markdown: effectiveMarkdown, images, decodedImages: fileResult.images ?? {}, pages: maxPage };
  }

  private generateMapping(result: MinerUResult, pdfPath: string): MappingData {
    return {
      version: 1,
      sourceFormat: 'application/pdf',
      sourcePath: pdfPath,
      mappings: [],
      figures: result.images.map((img, i) => ({
        id: `fig${i + 1}`,
        path: `assets/${img.filename}`,
        sourcePosition: {
          format: 'application/pdf',
          page: img.page,
          bbox: img.bbox,
        },
      })),
      equations: [],
      citations: [],
    };
  }

  private stringifyYaml(data: MappingData): string {
    const lines: string[] = [];
    lines.push(`version: ${data.version}`);
    lines.push(`sourceFormat: ${data.sourceFormat}`);
    lines.push(`sourcePath: ${JSON.stringify(data.sourcePath)}`);
    lines.push('mappings: []');
    lines.push('figures:');
    for (const f of data.figures) {
      lines.push(`  - id: ${f.id}`);
      lines.push(`    path: ${JSON.stringify(f.path)}`);
      lines.push(`    sourcePosition:`);
      lines.push(`      format: ${f.sourcePosition.format}`);
      if (f.sourcePosition.format === 'application/pdf') {
        const pos = f.sourcePosition as { page: number; bbox: [number, number, number, number] };
        lines.push(`      page: ${pos.page}`);
        lines.push(`      bbox: [${pos.bbox.join(', ')}]`);
      }
    }
    lines.push('equations: []');
    lines.push('citations: []');
    return lines.join('\n');
  }
}

export default {
  name: '@yandu/plugin-converter-pdf',
  version: '1.0.0',
  register(system) {
    const converter = new PDFConverter();
    system.capabilities.register(
      { type: 'converter', id: converter.id, name: 'MinerU PDF Converter' },
      converter,
    );
  },
} satisfies Plugin;
