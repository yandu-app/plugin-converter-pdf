#!/usr/bin/env node
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
const MAX_LINE_BYTES = 1024 * 1024, MAX_OUTPUT_BYTES = 4 * 1024 * 1024, MAX_ARGS = 32, TIMEOUT_MS = 30_000;
const fail = (id: string, code: string, message: string) => ({ id, ok: false as const, error: { code, message } });
export function validateRequest(value: any): any {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('request must be an object');
  if (Object.keys(value).some((key) => !['id','method','params'].includes(key))) throw new Error('request contains unknown fields');
  if (typeof value.id !== 'string' || value.id.length < 1 || value.id.length > 128) throw new Error('id must be a 1-128 character string');
  if (value.method !== 'invoke') throw new Error('method must be invoke');
  const params = value.params;
  if (!params || typeof params !== 'object' || Array.isArray(params)) throw new Error('params must be an object');
  if (Object.keys(params).some((key) => !['operation','args','config'].includes(key))) throw new Error('params contains unknown fields');
  if (typeof params.operation !== 'string' || !/^[A-Za-z][A-Za-z0-9]*$/.test(params.operation)) throw new Error('params.operation is invalid');
  if (!Array.isArray(params.args) || params.args.length > MAX_ARGS) throw new Error(`params.args must contain at most ${MAX_ARGS} values`);
  if (params.config !== undefined && (!params.config || typeof params.config !== 'object' || Array.isArray(params.config))) throw new Error('params.config must be an object');
  return value;
}
export async function handleRequest(value: any, load: () => Promise<any> = () => import('./index.js')): Promise<any> {
  let request: any; try { request = validateRequest(value); } catch (error) { return fail(typeof value?.id === 'string' ? value.id : '', 'INVALID_REQUEST', error instanceof Error ? error.message : String(error)); }
  try {
    let capability: any; const config = request.params.config ?? {};
    const runtime: any = { config: { get: (key: string) => config[key] }, logger: { debug() {}, info() {}, warn() {}, error() {} }, capabilities: { register: (_descriptor: unknown, implementation: unknown) => { capability = implementation; } }, getModel: () => { throw new Error('host model capability is unavailable in process mode'); } };
    const module = await load(); await module.default.register(runtime);
    if (!capability) return fail(request.id, 'NOT_CONFIGURED', 'plugin did not register a capability; check required configuration');
    const operation = capability[request.params.operation];
    if (typeof operation !== 'function') return fail(request.id, 'UNSUPPORTED_OPERATION', `unsupported operation: ${request.params.operation}`);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`operation exceeded ${TIMEOUT_MS} ms`)), TIMEOUT_MS); });
    const result = await Promise.race([operation.apply(capability, request.params.args), timeout]); clearTimeout(timer);
    const response = { id: request.id, ok: true as const, result };
    if (Buffer.byteLength(JSON.stringify(response)) > MAX_OUTPUT_BYTES) return fail(request.id, 'OUTPUT_TOO_LARGE', 'response exceeds 4 MiB');
    return response;
  } catch (error) { return fail(request.id, 'OPERATION_FAILED', error instanceof Error ? error.message : String(error)); }
}
async function main() { const lines = createInterface({ input: process.stdin, crlfDelay: Infinity }); for await (const line of lines) { let response; if (Buffer.byteLength(line) > MAX_LINE_BYTES) response = fail('', 'REQUEST_TOO_LARGE', 'request exceeds 1 MiB'); else try { response = await handleRequest(JSON.parse(line)); } catch (error) { response = fail('', 'INVALID_JSON', error instanceof Error ? error.message : String(error)); } process.stdout.write(`${JSON.stringify(response)}
`); } }
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) void main();
