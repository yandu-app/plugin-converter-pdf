import test from 'node:test';
import assert from 'node:assert/strict';
import { handleRequest, validateRequest } from '../dist/worker.js';
test('rejects unknown fields and invalid operation names', () => { assert.throws(() => validateRequest({id:'x',method:'invoke',params:{operation:'run',args:[],extra:true}}), /unknown/); assert.throws(() => validateRequest({id:'x',method:'invoke',params:{operation:'../run',args:[]}}), /invalid/); });
test('invokes only the registered capability', async () => { const load=async()=>({default:{register(r){r.capabilities.register({}, {run:(value)=>({value})});}}}); assert.deepEqual(await handleRequest({id:'r1',method:'invoke',params:{operation:'run',args:['ok']}},load), {id:'r1',ok:true,result:{value:'ok'}}); const missing=await handleRequest({id:'r2',method:'invoke',params:{operation:'missing',args:[]}},load); assert.equal(missing.error.code,'UNSUPPORTED_OPERATION'); });
