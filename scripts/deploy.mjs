// Build and publish a static Pages branch without touching the source index.
import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
const run=(bin,args,opts={})=>{const value=execFileSync(bin,args,{encoding:'utf8',...opts});return value==null?'':String(value).trim()};
const root=process.cwd();
run('npm',['test'],{stdio:'inherit'});
run('npm',['run','build'],{stdio:'inherit'});
const repo=JSON.parse(run('gh',['repo','view','--json','nameWithOwner'])).nameWithOwner;
const tmp=mkdtempSync(join(tmpdir(),'flash-atlas-deploy-'));
try{
 writeFileSync(join(root,'dist','.nojekyll'),'');
 const gitArgs=['--git-dir',join(root,'.git'),'--work-tree',join(root,'dist')];
 const env={...process.env,GIT_INDEX_FILE:join(tmp,'index')};
 const git=(...args)=>run('git',[...gitArgs,...args],{env,cwd:join(root,'dist')});
 let parent='';
 const remote=run('git',['ls-remote','--heads','origin','gh-pages']);
 if(remote){run('git',['fetch','origin','gh-pages']);parent=run('git',['rev-parse','FETCH_HEAD'])}
 git('read-tree','--empty');git('add','--all','--force','.');const tree=git('write-tree');
 const source=run('git',['rev-parse','--short','HEAD']);
 const commit=git('commit-tree',tree,...(parent?['-p',parent]:[]),'-m',`Deploy Flash Atlas from ${source}`);
 run('git',['push','origin',`${commit}:refs/heads/gh-pages`],{stdio:'inherit'});
 let page;
 try{page=JSON.parse(run('gh',['api',`repos/${repo}/pages`]))}
 catch{page=JSON.parse(run('gh',['api','--method','POST',`repos/${repo}/pages`,'--input','-'],{input:JSON.stringify({source:{branch:'gh-pages',path:'/'}})}))}
 if(page.source?.branch!=='gh-pages')run('gh',['api','--method','PUT',`repos/${repo}/pages`,'--input','-'],{input:JSON.stringify({source:{branch:'gh-pages',path:'/'}})});
 console.log(`Pages: ${page.html_url}`);
}finally{rmSync(tmp,{recursive:true,force:true})}
