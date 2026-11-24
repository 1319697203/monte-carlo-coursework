"""课程题目1：Beta-PERT任务采样、动态CPM与项目进度风险分析。

输入为教学假设数据；本程序输出不是任何实际项目的实测结果。
运行：python src/simulate.py --n 50000 --seed 20261024 --deadline 35
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


def load_tasks(path):
    """检查工期、依赖，并用拓扑排序允许CSV行顺序任意。"""
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('任务表为空')
    tasks = {}
    for r in rows:
        key = r['id'].strip()
        if not key or key in tasks:
            raise ValueError('任务ID为空或重复')
        r['id'] = key
        r['preds'] = [p.strip() for p in r['preds'].split(';') if p.strip()]
        for k in ('a', 'm', 'b'):
            r[k] = float(r[k])
        if not all(np.isfinite(r[k]) for k in ('a','m','b')) or not 0 <= r['a'] <= r['m'] <= r['b']:
            raise ValueError(f'{key}: 要求0<=a<=m<=b且为有限数')
        tasks[key] = r
    for r in rows:
        if any(p not in tasks for p in r['preds']):
            raise ValueError('存在未知前置任务')
    ordered = []
    pending = list(tasks)
    while pending:
        ready = [k for k in pending if all(p in {r['id'] for r in ordered} for p in tasks[k]['preds'])]
        if not ready:
            raise ValueError('任务依赖存在环')
        for k in ready:
            ordered.append(tasks[k]); pending.remove(k)
    return ordered


def graph(tasks):
    ids = {r['id']: i for i, r in enumerate(tasks)}
    pred = [[ids[p] for p in r['preds']] for r in tasks]
    succ = [[] for _ in tasks]
    for i, ps in enumerate(pred):
        for p in ps:
            succ[p].append(i)
    return pred, succ


def pert_parameters(a, m, b, weight=4.0):
    """非退化任务的Beta形状参数；退化任务在采样函数中直接赋常数。"""
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('PERT权重必须为正的有限数')
    width = b-a
    safe = np.where(width > 0, width, 1.0)
    return 1+weight*(m-a)/safe, 1+weight*(b-m)/safe


def sample_durations(tasks, n, seed, weight=4.0):
    a, m, b = (np.array([r[k] for r in tasks]) for k in ('a','m','b'))
    alpha, beta = pert_parameters(a, m, b, weight)
    rng = np.random.default_rng(seed)
    # 每一行是一轮完整项目；任务工期在基准模型中相互独立。
    return a + (b-a)*rng.beta(alpha, beta, size=(n, len(tasks)))


def schedule(durations, pred, succ):
    """按拓扑序前推/后推；每一轮重算，不能固定某一条关键路径。

    以公共零工期终点处理多个末端任务，允许同一轮存在多条关键路径。
    """
    durations = np.atleast_2d(np.asarray(durations, dtype=float))
    es = np.zeros_like(durations)
    ef = np.zeros_like(durations)
    for i, ps in enumerate(pred):
        if ps:
            es[:, i] = ef[:, ps].max(axis=1)
        ef[:, i] = es[:, i] + durations[:, i]
    total = ef.max(axis=1)
    ls = np.zeros_like(durations)
    lf = np.zeros_like(durations)
    for i in range(len(pred)-1, -1, -1):
        lf[:, i] = ls[:, succ[i]].min(axis=1) if succ[i] else total
        ls[:, i] = lf[:, i] - durations[:, i]
    slack = ls-es
    critical = np.isclose(slack, 0.0, atol=1e-8, rtol=0)
    return total, es, ef, ls, lf, slack, critical


def wilson(k, n, z=1.959963984540054):
    p = k/n
    c = (p+z*z/(2*n))/(1+z*z/n)
    h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(c-h), float(c+h)]


def summarize(total, deadline):
    n = len(total)
    k = int((total <= deadline).sum())
    return {'n': n, 'mean': float(total.mean()), 'std': float(total.std(ddof=1)),
            'mean_mc_se': float(total.std(ddof=1)/np.sqrt(n)),
            'p50': float(np.quantile(total,.5)), 'p80': float(np.quantile(total,.8)),
            'p90': float(np.quantile(total,.9)), 'p95': float(np.quantile(total,.95)),
            'deadline': deadline, 'on_time_count': k, 'on_time_probability': k/n,
            'on_time_wilson95': wilson(k,n)}


def csv_write(path, fields, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def plots(tasks, results, durations, total, sensitivity, baseline, scenarios, out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    navy, teal, orange = '#17324D','#147D92','#D57932'
    figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    def save(name):
        plt.savefig(figdir/(name+'.png'), dpi=180, bbox_inches='tight')
        plt.savefig(figdir/(name+'.svg'), bbox_inches='tight')
        plt.close()
    fig, ax=plt.subplots(figsize=(8.2,4.5))
    ax.hist(total,bins=55,color=teal,alpha=.85,edgecolor='white',linewidth=.3)
    for x,c,l in [(results['cpm_m'],navy,'CPM (m)'),(results['baseline']['mean'],orange,'MC mean'),(results['deadline'],'#B74444','Deadline')]:
        ax.axvline(x,color=c,linestyle='--',label=f'{l}: {x:.2f}')
    ax.set(xlabel='Project duration (working days)',ylabel='Simulation count',title='Project duration distribution'); ax.legend()
    save('02_duration_histogram')
    fig, ax=plt.subplots(figsize=(8.2,4.5))
    ordered=np.sort(total); ax.plot(ordered,np.arange(1,len(total)+1)/len(total),color=teal)
    ax.axvline(results['deadline'],color=orange,linestyle='--',label='Deadline')
    for q in (.8,.9):
        x=float(np.quantile(total,q)); ax.plot(x,q,'o',color=navy); ax.annotate(f'P{int(100*q)} = {x:.2f}',(x,q),xytext=(8,-20 if q==.8 else 8),textcoords='offset points')
    ax.set(xlabel='Project duration (working days)',ylabel='Completion probability',ylim=(0,1.04),title='Empirical cumulative distribution'); ax.grid(alpha=.2); ax.legend()
    save('03_completion_cdf')
    ordered_s=sorted(sensitivity,key=lambda r:r['spearman'])
    fig, axs=plt.subplots(1,2,figsize=(9,5.5),sharey=True)
    ids=[r['id'] for r in ordered_s]; y=np.arange(len(ids))
    axs[0].barh(y,[r['spearman'] for r in ordered_s],color=teal); axs[0].set_yticks(y,ids); axs[0].set(xlabel='Spearman correlation',title='Duration sensitivity')
    axs[1].barh(y,[100*r['criticality'] for r in ordered_s],color=orange); axs[1].set(xlabel='Criticality index (%)',title='Frequency on a critical path',xlim=(0,105))
    fig.tight_layout(); save('04_sensitivity')
    # 网络图沿纵向分层，节点用ID，中文含义见WBS表，避免系统字体差异。
    pred,succ=graph(tasks); level=[]
    for i,ps in enumerate(pred): level.append(0 if not ps else max(level[p] for p in ps)+1)
    coords={}
    for layer in sorted(set(level)):
        members=[i for i,l in enumerate(level) if l==layer]
        for k,i in enumerate(members): coords[i]=((k-(len(members)-1)/2)*2.1,-layer)
    fig,ax=plt.subplots(figsize=(7,8))
    critical=baseline[-1][0]
    for i,ps in enumerate(pred):
        for p in ps:
            edge_critical=critical[p] and critical[i] and abs(baseline[2][0,p]-baseline[1][0,i])<1e-8
            ax.add_patch(FancyArrowPatch(coords[p],coords[i],arrowstyle='-|>',mutation_scale=12,shrinkA=21,shrinkB=21,color=orange if edge_critical else '#ABB5BF',linewidth=1.7 if edge_critical else 1.1))
    for i,r in enumerate(tasks):
        ax.text(*coords[i],f"{r['id']}\nm={r['m']:g}",ha='center',va='center',bbox=dict(boxstyle='round,pad=.45',fc='#FFF1E6' if critical[i] else '#E7F2F4',ec=orange if critical[i] else teal),fontsize=10)
    xs=[p[0] for p in coords.values()]; ax.set(xlim=(min(xs)-1,max(xs)+1),ylim=(-max(level)-.8,.8),title='Task dependency network | orange: CPM critical path'); ax.axis('off')
    save('01_task_network')
    fig,ax=plt.subplots(figsize=(8.2,5.5))
    es=baseline[1][0]; ds=baseline[2][0]-es
    ax.barh(np.arange(len(tasks)),ds,left=es,color=[orange if c else teal for c in critical])
    ax.set_yticks(np.arange(len(tasks)),[r['id'] for r in tasks]); ax.invert_yaxis(); ax.set(xlabel='Working day from project start',title='Deterministic CPM schedule (m)'); ax.grid(axis='x',alpha=.2)
    save('05_cpm_gantt')
    fig,axs=plt.subplots(1,2,figsize=(9,4.4))
    names=list(scenarios); labels=['Baseline','Mitigation','Wider tail']
    axs[0].bar(labels,[scenarios[k]['mean'] for k in names],color=[teal,navy,orange]); axs[0].set(ylabel='Working days',title='Scenario mean duration')
    axs[1].bar(labels,[100*scenarios[k]['on_time_probability'] for k in names],color=[teal,navy,orange]); axs[1].set(ylabel='Probability (%)',ylim=(0,100),title=f"Completion by day {results['deadline']:g}")
    fig.tight_layout(); save('06_scenario_comparison')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/tasks.csv')
    p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--n',type=int,default=50000)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--deadline',type=float,default=35)
    p.add_argument('--weight',type=float,default=4)
    args=p.parse_args()
    if args.n<10000: p.error('课程作业要求至少10000次模拟')
    if not np.isfinite(args.deadline) or args.deadline<0: p.error('截止工期必须为非负有限数')
    tasks=load_tasks(args.input); pred,succ=graph(tasks)
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    m=np.array([r['m'] for r in tasks])
    expected=np.array([(r['a']+args.weight*r['m']+r['b'])/(args.weight+2) for r in tasks])
    base=schedule(m,pred,succ); expbase=schedule(expected,pred,succ)
    durations=sample_durations(tasks,args.n,args.seed,args.weight)
    total,es,ef,ls,lf,slack,critical=schedule(durations,pred,succ)
    ranked_total=rankdata(total)
    sensitivities=[]
    for i,r in enumerate(tasks):
        rho=float(np.corrcoef(rankdata(durations[:,i]),ranked_total)[0,1]) if np.ptp(durations[:,i])>0 and np.ptp(total)>0 else 0.0
        sensitivities.append({'id':r['id'],'name':r['name'],'spearman':rho,'criticality':float(critical[:,i].mean())})
    baseline=summarize(total,args.deadline)
    # 管理情景是假设干预，不是观测到的真实效果；不据此承诺加人收益。
    mitigation=[dict(r) for r in tasks]
    changes={'E':8.,'J':6.,'M':5.}
    applied={}
    for r in mitigation:
        if r['id'] in changes and r['m']<=changes[r['id']]<r['b']:
            applied[r['id']]={'old_b':r['b'],'new_b':changes[r['id']]}; r['b']=changes[r['id']]
    wider=[dict(r,b=r['m']+1.2*(r['b']-r['m'])) for r in tasks]
    scenario_tasks={'mitigation':mitigation,'wider_tail':wider}
    scenarios={'baseline':baseline}
    scenario_totals={}
    for j,(name,ts) in enumerate(scenario_tasks.items(),1):
        draw=sample_durations(ts,args.n,args.seed+j,args.weight)
        t=schedule(draw,pred,succ)[0]; scenario_totals[name]=t; scenarios[name]=summarize(t,args.deadline)
    results={'project':'校园报修管理系统','data_status':'教学假设案例，非实测数据','n':args.n,'seed':args.seed,'deadline':args.deadline,'weight':args.weight,
             'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
             'cpm_m':float(base[0][0]),'cpm_expected':float(expbase[0][0]),
             'cpm_critical_ids':[r['id'] for i,r in enumerate(tasks) if base[-1][0,i]],
             'probability_at_cpm_m':float((total<=base[0][0]).mean()),
             'probability_at_cpm_expected':float((total<=expbase[0][0]).mean()),
             'baseline':baseline,'scenarios':scenarios,'mitigation_changes':applied,
             'scenario_seeds':{'baseline':args.seed,'mitigation':args.seed+1,'wider_tail':args.seed+2},
             'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}}
    (out/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    csv_write(out/'sensitivity.csv',list(sensitivities[0]),sorted(sensitivities,key=lambda r:-r['spearman']))
    rows=[]
    for i,r in enumerate(tasks):
        rows.append({'id':r['id'],'name':r['name'],'expected_duration':expected[i],'ES':base[1][0,i],'EF':base[2][0,i],'LS':base[3][0,i],'LF':base[4][0,i],'total_float':base[5][0,i],'critical':bool(base[6][0,i])})
    csv_write(out/'cpm_schedule.csv',list(rows[0]),rows)
    probability_rows=[]
    for d in sorted(set([float(x) for x in range(28,43)]+[args.deadline,float(base[0][0]),float(expbase[0][0])])):
        k=int((total<=d).sum()); ci=wilson(k,args.n)
        probability_rows.append({'deadline':d,'probability':k/args.n,'wilson95_low':ci[0],'wilson95_high':ci[1]})
    csv_write(out/'deadline_probabilities.csv',list(probability_rows[0]),probability_rows)
    convergence=[]
    for n in sorted(set([10000,args.n//2,args.n])):
        if n<=args.n:
            s=summarize(total[:n],args.deadline)
            convergence.append({k:s[k] for k in ['n','mean','p90','on_time_probability']})
    csv_write(out/'convergence.csv',list(convergence[0]),convergence)
    # 保存原始数值，避免只留下无法复核的图；float64保持原始精度。
    np.savez_compressed(out/'simulation_samples.npz',durations=durations,total=total,critical=critical,task_ids=np.array([r['id'] for r in tasks]),**scenario_totals)
    plots(tasks,results,durations,total,sensitivities,base,scenarios,out)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
