import React, { useState, useEffect, useMemo } from 'react';
import {
    Button, Drawer, Select, MultiSelect, TextInput, Group, Notification, Title, Divider, SegmentedControl, Text, Flex
} from '@mantine/core';
import { IconBrain, IconCheck, IconX } from '@tabler/icons-react';

const ALL_ALGORITHMS = [
    // classifiers
    { value: 'random_forest',       label: 'Random Forest',              type: 'classifier',  supportsSeq: false },
    { value: 'gradient_boosting',   label: 'Gradient Boosting',          type: 'classifier',  supportsSeq: false },
    { value: 'adaboost',            label: 'AdaBoost',                   type: 'classifier',  supportsSeq: false },
    { value: 'extra_trees',         label: 'ExtraTrees',                 type: 'classifier',  supportsSeq: false },
    { value: 'decision_tree',       label: 'Decision Tree',              type: 'classifier',  supportsSeq: false },
    { value: 'logistic_regression', label: 'Logistic Regression',        type: 'classifier',  supportsSeq: false },
    { value: 'ridge_classifier',    label: 'Ridge Classifier',           type: 'classifier',  supportsSeq: false },
    { value: 'svm',                 label: 'Support Vector Machine',     type: 'classifier',  supportsSeq: false },
    { value: 'knn',                 label: 'K-Nearest Neighbors',        type: 'classifier',  supportsSeq: false },
    { value: 'gaussian_nb',         label: 'GaussianNB',                 type: 'classifier',  supportsSeq: false },
    { value: 'mlp',                 label: 'MLP Classifier',             type: 'classifier',  supportsSeq: true  }, // supports multi-output

    // regressors
    { value: 'random_forest_reg',     label: 'Random Forest Regressor',     type: 'regressor', supportsSeq: false },
    { value: 'gradient_boosting_reg', label: 'Gradient Boosting Regressor', type: 'regressor', supportsSeq: false },
    { value: 'adaboost_reg',          label: 'AdaBoost Regressor',          type: 'regressor', supportsSeq: false },
    { value: 'extra_trees_reg',       label: 'ExtraTrees Regressor',        type: 'regressor', supportsSeq: false },
    { value: 'decision_tree_reg',     label: 'Decision Tree Regressor',     type: 'regressor', supportsSeq: false },
    { value: 'linear_regression',     label: 'Linear Regression',           type: 'regressor', supportsSeq: false },
    { value: 'ridge',                 label: 'Ridge Regression',            type: 'regressor', supportsSeq: false },
    { value: 'lasso',                 label: 'Lasso',                       type: 'regressor', supportsSeq: false },
    { value: 'elastic_net',           label: 'ElasticNet',                  type: 'regressor', supportsSeq: false },
    { value: 'svr',                   label: 'Support Vector Regressor',    type: 'regressor', supportsSeq: false },
    { value: 'knn_reg',               label: 'KNN Regressor',               type: 'regressor', supportsSeq: false },
    { value: 'mlp_reg',               label: 'MLP Regressor',               type: 'regressor', supportsSeq: true  }, // supports multi-output
] as const;

type Algorithm = typeof ALL_ALGORITHMS[number]['value'];
type Mode = 'classic' | 'sliding';
type Tuning  = 'manual' | 'random' | 'grid' | 'halving' | 'evolutionary';

interface DataSource { id: string; name: string }
interface Snapshot   { id: string; created_at: string }
interface ColumnDef  { name: string; dtype: string }
interface ColumnSummary {
    column: string;
    type: 'numeric' | 'categorical' | 'date';
    missing: number;
    missing_pct: number;
    unique: number;
    stats: string;
}
interface WindowFeature { name: string; start_idx: number; end_idx: number }
interface WindowSpec    { features: WindowFeature[]; target: WindowFeature }

interface Props { onCreate(): void }

export function TrainingForm({ onCreate }: Props) {
    // --- State ---
    const [open, setOpen] = useState(false);
    const [mode, setMode] = useState<Mode>('classic');
    const [datasources, setDatasources] = useState<DataSource[]>([]);
    const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
    const [schemaCols, setSchemaCols] = useState<ColumnDef[]>([]);
    const [serverSummary, setServerSummary] = useState<ColumnSummary[]>([]);
    const [loadingSummary, setLoadingSummary] = useState(false);

    const [name, setName] = useState('');
    const [datasourceId, setDatasourceId] = useState<string | null>(null);
    const [snapshotId, setSnapshotId] = useState<string | null>(null);

    // Classic mode
    const [features, setFeatures] = useState<string[]>([]);
    const [target, setTarget] = useState<string | null>(null);

    // Sliding Window mode
    const [windowSpec, setWindowSpec] = useState<WindowSpec>({
        features: [],
        target: { name: '', start_idx: 0, end_idx: 0 }
    });

    const [algorithm, setAlgorithm] = useState<Algorithm>('random_forest');
    const [tuning, setTuning] = useState<Tuning>('manual');
    const [params, setParams] = useState<Record<string, any>>({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);
    const [previewing, setPreviewing] = useState(false);
    const [previewOK, setPreviewOK] = useState(false);
    const [previewError, setPreviewError] = useState<string | null>(null);


    // Tuning state
    const [hpoIters, setHpoIters] = useState(20);
    const [hpoCv, setHpoCv] = useState(3);
    const [hpoFactor, setHpoFactor] = useState(3);
    const [hpoMinResources, setHpoMinResources] = useState(1);
    const [hpoPop, setHpoPop] = useState(20);
    const [hpoGens, setHpoGens] = useState(10);

    type SplitMethod = 'random' | 'stratified' | 'time_series' | 'kfold';
    const [splitMethod, setSplitMethod] = useState<SplitMethod>('random');
    const [splitRatio, setSplitRatio] = useState<number>(0.2);

    // --- Load datasources ---
    useEffect(() => {
        if (!open) return;
        fetch('http://localhost:8000/datasources/')
            .then(r => r.json())
            .then(setDatasources)
            .catch(console.error);
    }, [open]);

    // --- Load snapshots+schema ---
    useEffect(() => {
        if (!datasourceId) return;
        fetch(`http://localhost:8000/datasources/${datasourceId}/with-snapshot/`)
            .then(r => r.json())
            .then(ds => {
                setSnapshots(ds.snapshots || []);
                if (ds.snapshots?.length) setSnapshotId(ds.snapshots[0].id);
                try {
                    const cols = JSON.parse(ds.schema_json || '{"columns":[]}').columns;
                    setSchemaCols(cols);
                } catch {
                    setSchemaCols([]);
                }
            })
            .catch(console.error);
    }, [datasourceId]);

    // --- Load column summary for target/sequence detection ---
    useEffect(() => {
        if (!datasourceId || !snapshotId) return;
        setLoadingSummary(true);
        fetch(`http://localhost:8000/datasources/${datasourceId}/snapshots/${snapshotId}/summary/`)
            .then(r => r.json())
            .then((json: { summary: ColumnSummary[] }) => {
                setServerSummary(json.summary);
                setLoadingSummary(false);
            })
            .catch(() => setLoadingSummary(false));
    }, [datasourceId, snapshotId]);

    // --- Sequence output length (for sliding window) ---
    const seqOutputLen = useMemo(() => {
        if (mode !== 'sliding') return 1;
        const { start_idx, end_idx } = windowSpec.target;
        return Math.abs(end_idx - start_idx) + 1;
    }, [mode, windowSpec.target]);

    // --- Infer task type (for both modes) ---
    const inferredTaskType = useMemo(() => {
        let tcol: string | undefined;
        if (mode === 'classic') tcol = target ?? undefined;
        else if (mode === 'sliding') tcol = windowSpec.target.name || undefined;
        if (!tcol) return null;
        const col = serverSummary.find(c => c.column === tcol);
        if (!col) return null;
        if (col.type === 'numeric') {
            if (col.unique <= 20) return 'classification';
            return 'regression';
        }
        if (col.type === 'categorical') return 'classification';
        return null;
    }, [mode, target, windowSpec.target, serverSummary]);

    // --- Algorithm filtering logic ---
    const algorithmOptions = useMemo(() => {
        if (!inferredTaskType) return [];
        // For sliding window, only show supportsSeq=true if seqOutputLen > 1
        return ALL_ALGORITHMS
            .filter(a =>
                a.type === (inferredTaskType === 'classification' ? 'classifier' : 'regressor')
                && (mode === 'classic' || seqOutputLen <= 1 || a.supportsSeq)
            )
            .map(opt => ({ value: opt.value, label: opt.label }));
    }, [inferredTaskType, mode, seqOutputLen]);

    // --- Auto-pick algorithm if necessary
    useEffect(() => {
        if (algorithmOptions.length && (!algorithmOptions.some(a => a.value === algorithm) || !algorithm)) {
            setAlgorithm(algorithmOptions[0].value as Algorithm);
        }
    }, [algorithmOptions.length, inferredTaskType, mode, seqOutputLen]);

    // --- completeness checks ---
    const incompleteClassic = mode === 'classic' && (!features.length || !target);
    const incompleteSliding = mode === 'sliding' && (
        !windowSpec.features.length ||
        !windowSpec.target.name
    );
    const disablePreview = !snapshotId || (mode === 'classic' ? incompleteClassic : incompleteSliding);


    // --- assemble config payload ---
    const config = useMemo(() => {
        let base: any = {
            algorithm,
            params,
            split_method: splitMethod,
            split_ratio: splitRatio
        };

        // Features/Target logic
        if (mode === 'classic') {
            base = { ...base, features, target };
        } else {
            base = { ...base, window_spec: windowSpec };
        }

        // Tuning logic
        if (tuning !== 'manual') {
            base.hpo = true;
            base.hpo_strategy = tuning;
            base.hpo_cv = hpoCv;
        }
        if (tuning === 'random') {
            base.hpo_iters = hpoIters;
        }
        if (tuning === 'halving') {
            base.hpo_factor = hpoFactor;
            base.hpo_min_resources = hpoMinResources;
        }
        if (tuning === 'evolutionary') {
            base.hpo_pop = hpoPop;
            base.hpo_gen = hpoGens;
        }
        if (splitMethod === 'kfold') {
            base.kfold = params.kfold ?? 5;
        }

        return base;
    }, [
        algorithm, params, splitMethod, splitRatio,
        features, target, windowSpec, mode, tuning,
        hpoCv, hpoIters, hpoFactor, hpoMinResources, hpoPop, hpoGens
    ]);

    // 7) preview
    const handlePreview = async () => {
        setPreviewError(null);
        setPreviewOK(false);
        setPreviewing(true);
        try {
            const resp = await fetch('http://localhost:8000/trainings/preview/', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ snapshot_id: snapshotId, config }),
            });
            const jd = await resp.json();
            if (!resp.ok) throw new Error(jd.detail || 'Preview failed');
            setPreviewOK(true);
        } catch(e:any) {
            setPreviewError(e.message);
        } finally {
            setPreviewing(false);
        }
    };

    // 8) train
    const handleTrain = async () => {
        setLoading(true);
        setError(null);
        setSuccess(false);
        try {
            const create = await fetch('http://localhost:8000/trainings/', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ name, datasource_id: datasourceId, config }),
            });
            if (!create.ok) {
                const err = await create.json();
                throw new Error(err.detail || `Status ${create.status}`);
            }
            const tr = await create.json();
            const exec = await fetch(`http://localhost:8000/trainings/${tr.id}/execute/`, {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ snapshot_id: snapshotId }),
            });
            if (!exec.ok) {
                const ed = await exec.json();
                throw new Error(ed.detail || 'Execute failed');
            }
            setSuccess(true);
            onCreate();
        } catch(e:any) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    // Notification for Ray
    const rayAlgos = [
        'xgboost_ray_cls', 'xgboost_ray_reg', 'lightgbm_ray_cls', 'lightgbm_ray_reg'
    ];

    // 9) hyperparameter UI
    const renderHyperparams = () => {
        switch(algorithm) {
            // — RandomForest —
            case 'random_forest':
            case 'random_forest_reg':
                return <>
                    <TextInput
                        label="n_estimators"
                        type="number"
                        value={String(params.n_estimators ?? 100)}
                        onChange={e => setParams(p=>({...p, n_estimators: +e.currentTarget.value}))}
                    />
                    <TextInput
                        label="max_depth"
                        type="number"
                        value={params.max_depth != null ? String(params.max_depth) : ''}
                        onChange={e => {
                            const v = e.currentTarget.value;
                            setParams(p=>({...p, max_depth: v ? +v : undefined}));
                        }}
                    />
                </>;

            // — GradientBoosting —
            case 'gradient_boosting':
            case 'gradient_boosting_reg':
                return <>
                    <TextInput
                        label="n_estimators"
                        type="number"
                        value={String(params.n_estimators ?? 100)}
                        onChange={e=>setParams(p=>({...p, n_estimators:+e.currentTarget.value}))}
                    />
                    <TextInput
                        label="learning_rate"
                        type="number"
                        step="0.01"
                        value={String(params.learning_rate ?? 0.1)}
                        onChange={e=>setParams(p=>({...p, learning_rate:+e.currentTarget.value}))}
                    />
                    <TextInput
                        label="max_depth"
                        type="number"
                        value={String(params.max_depth ?? 3)}
                        onChange={e=>setParams(p=>({...p, max_depth:+e.currentTarget.value}))}
                    />
                </>;

            // — AdaBoost —
            case 'adaboost':
            case 'adaboost_reg':
                return <>
                    <TextInput
                        label="n_estimators"
                        type="number"
                        value={String(params.n_estimators ?? 50)}
                        onChange={e=>setParams(p=>({...p, n_estimators:+e.currentTarget.value}))}
                    />
                    <TextInput
                        label="learning_rate"
                        type="number"
                        step="0.01"
                        value={String(params.learning_rate ?? 1)}
                        onChange={e=>setParams(p=>({...p, learning_rate:+e.currentTarget.value}))}
                    />
                </>;

            // — ExtraTrees —
            case 'extra_trees':
            case 'extra_trees_reg':
                return <>
                    <TextInput
                        label="n_estimators"
                        type="number"
                        value={String(params.n_estimators ?? 100)}
                        onChange={e=>setParams(p=>({...p, n_estimators:+e.currentTarget.value}))}
                    />
                    <TextInput
                        label="max_depth"
                        type="number"
                        value={String(params.max_depth ?? undefined)}
                        onChange={e=>setParams(p=>({...p, max_depth:e.currentTarget.value?+e.currentTarget.value:undefined}))}
                    />
                </>;

            // — DecisionTree —
            case 'decision_tree':
            case 'decision_tree_reg':
                return <>
                    <TextInput
                        label="max_depth"
                        type="number"
                        value={String(params.max_depth ?? undefined)}
                        onChange={e=>setParams(p=>({...p, max_depth:e.currentTarget.value?+e.currentTarget.value:undefined}))}
                    />
                    <TextInput
                        label="min_samples_split"
                        type="number"
                        value={String(params.min_samples_split ?? 2)}
                        onChange={e=>setParams(p=>({...p, min_samples_split:+e.currentTarget.value}))}
                    />
                </>;

            // — Linear / Ridge / Lasso / ElasticNet —
            case 'logistic_regression':
                return <TextInput
                    label="C"
                    type="number"
                    step="0.01"
                    value={String(params.C ?? 1)}
                    onChange={e=>setParams(p=>({...p, C:+e.currentTarget.value}))}
                />;
            case 'ridge_classifier':
                return <TextInput
                    label="alpha"
                    type="number"
                    step="0.1"
                    value={String(params.alpha ?? 1)}
                    onChange={e=>setParams(p=>({...p, alpha:+e.currentTarget.value}))}
                />;
            case 'linear_regression':
                return null;
            case 'ridge':
            case 'lasso':
                return <TextInput
                    label="alpha"
                    type="number"
                    step="0.1"
                    value={String(params.alpha ?? 1)}
                    onChange={e=>setParams(p=>({...p, alpha:+e.currentTarget.value}))}
                />;
            case 'elastic_net':
                return <>
                    <TextInput
                        label="alpha"
                        type="number"
                        step="0.1"
                        value={String(params.alpha ?? 1)}
                        onChange={e=>setParams(p=>({...p, alpha:+e.currentTarget.value}))}
                    />
                    <TextInput
                        label="l1_ratio"
                        type="number"
                        step="0.05"
                        value={String(params.l1_ratio ?? 0.5)}
                        onChange={e=>setParams(p=>({...p, l1_ratio:+e.currentTarget.value}))}
                    />
                </>;

            // — SVM / SVR —
            case 'svm':
            case 'svr':
                return <>
                    <TextInput
                        label="kernel"
                        value={params.kernel ?? 'rbf'}
                        onChange={e=>setParams(p=>({...p, kernel:e.currentTarget.value}))}
                    />
                    <TextInput
                        label="C"
                        type="number"
                        step="0.1"
                        value={String(params.C ?? 1)}
                        onChange={e=>setParams(p=>({...p, C:+e.currentTarget.value}))}
                    />
                    {algorithm === 'svr' && (
                        <TextInput
                            label="epsilon"
                            type="number"
                            step="0.01"
                            value={String(params.epsilon ?? 0.1)}
                            onChange={e=>setParams(p=>({...p, epsilon:+e.currentTarget.value}))}
                        />
                    )}
                </>;

            // — KNN —
            case 'knn':
            case 'knn_reg':
                return <TextInput
                    label="n_neighbors"
                    type="number"
                    value={String(params.n_neighbors ?? 5)}
                    onChange={e=>setParams(p=>({...p, n_neighbors:+e.currentTarget.value}))}
                />;

            // — GaussianNB —
            case 'gaussian_nb':
                return <TextInput
                    label="var_smoothing"
                    type="number"
                    step="1e-9"
                    value={String(params.var_smoothing ?? 1e-9)}
                    onChange={e=>setParams(p=>({...p, var_smoothing:+e.currentTarget.value}))}
                />;

            // — MLPClassifier / MLPRegressor —
            case 'mlp':
            case 'mlp_reg':
                return <>
                    <TextInput
                        label="hidden_layer_sizes"
                        placeholder="e.g. 100,50"
                        value={Array.isArray(params.hidden_layer_sizes)
                            ? (params.hidden_layer_sizes as number[]).join(',')
                            : ''}
                        onChange={e=>{
                            const arr = e.currentTarget.value
                                .split(',')
                                .map(s=>parseInt(s,10))
                                .filter(n=>!isNaN(n));
                            setParams(p=>({...p, hidden_layer_sizes: arr}));
                        }}
                    />
                    <TextInput
                        label="alpha"
                        type="number"
                        step="0.0001"
                        value={String(params.alpha ?? 0.0001)}
                        onChange={e=>setParams(p=>({...p, alpha:+e.currentTarget.value}))}
                    />
                    <TextInput
                        label="learning_rate_init"
                        type="number"
                        step="0.0001"
                        value={String(params.learning_rate_init ?? 0.001)}
                        onChange={e=>setParams(p=>({...p, learning_rate_init:+e.currentTarget.value}))}
                    />
                </>;
            default:
                return null;
        }
    };

    return (
        <>
            <Button onClick={() => setOpen(true)} h="auto" w="100%" pt="15px" pb="15px" variant="default" color="#D0B9F0" c="#D0B9F0">
                <Flex direction="column" align="center"><IconBrain size={24}/><Text>Training</Text></Flex>
            </Button>
            <Drawer opened={open} onClose={() => setOpen(false)} size="50%" title="Train a Model">

                <Title order={4} mb="md">Configure Training</Title>

                {/* Name / Datasource / Snapshot */}
                <TextInput
                    label="Name"
                    value={name}
                    onChange={e => setName(e.currentTarget.value)}
                    mb="sm"
                />
                <Select
                    label="Datasource"
                    data={datasources.map(d => ({ value: d.id, label: d.name }))}
                    value={datasourceId || ''}
                    onChange={setDatasourceId}
                    mb="sm"
                />
                <Select
                    label="Snapshot"
                    data={snapshots.map(s => ({ value: s.id, label: new Date(s.created_at).toLocaleString() }))}
                    value={snapshotId || ''}
                    onChange={setSnapshotId}
                    disabled={!snapshots.length}
                    mb="md"
                />
                <Divider mb="md"></Divider>
                <SegmentedControl
                    value={mode}
                    onChange={v => setMode(v as Mode)}
                    data={[
                        { label: 'Classic', value: 'classic' },
                        { label: 'Sliding Window', value: 'sliding' }
                    ]}
                    mb="md"
                />

                {/* Classic */}
                {mode === 'classic' && (
                    <>
                        <MultiSelect
                            label="Features"
                            data={schemaCols.map(c=>({value:c.name,label:c.name}))}
                            value={features}
                            onChange={setFeatures}
                            mb="sm"
                        />
                        <Select
                            label="Target"
                            data={schemaCols.map(c=>({value:c.name,label:c.name}))}
                            value={target||''}
                            onChange={setTarget}
                            mb="md"
                            disabled={loadingSummary}
                        />
                    </>
                )}
                {/* Sliding Window */}
                {mode === 'sliding' && (
                    <>
                        <Divider my="sm"/><Title order={6}>Sliding-Window Spec</Title>
                        {windowSpec.features.map((f,i)=>(
                            <Group key={i} mb="xs">
                                <Select
                                    data={schemaCols.map(c=>({value:c.name,label:c.name}))}
                                    label="Feature" value={f.name}
                                    onChange={v=>setWindowSpec(ws=>({
                                        ...ws,
                                        features: ws.features.map((x,j)=> j===i
                                            ? {...x,name:v!} : x)
                                    }))}
                                />
                                <TextInput
                                    label="Start" type="number" style={{width:80}}
                                    value={String(f.start_idx)}
                                    onChange={e=>setWindowSpec(ws=>({
                                        ...ws,
                                        features: ws.features.map((x,j)=> j===i
                                            ? {...x,start_idx:+e.currentTarget.value||0} : x)
                                    }))}
                                />
                                <TextInput
                                    label="End" type="number" style={{width:80}}
                                    value={String(f.end_idx)}
                                    onChange={e=>setWindowSpec(ws=>({
                                        ...ws,
                                        features: ws.features.map((x,j)=> j===i
                                            ? {...x,end_idx:+e.currentTarget.value||0} : x)
                                    }))}
                                />
                            </Group>
                        ))}
                        <Button size="xs" mb="md"
                                onClick={()=>setWindowSpec(ws=>({
                                    ...ws,
                                    features:[...ws.features,{name:'',start_idx:0,end_idx:0}]
                                }))}
                        >+ Add Feature</Button>
                        <Divider my="sm"/>
                        <Select
                            label="Target"
                            data={schemaCols.map(c=>({value:c.name,label:c.name}))}
                            value={windowSpec.target.name}
                            onChange={v=>setWindowSpec(ws=>({
                                ...ws,
                                target:{...ws.target,name:v!}
                            }))}
                            mb="xs"
                        />
                        <Group>
                            <TextInput
                                label="Start" type="number" style={{width:80}}
                                value={String(windowSpec.target.start_idx)}
                                onChange={e=>setWindowSpec(ws=>({
                                    ...ws,
                                    target:{...ws.target,start_idx:+e.currentTarget.value||0}
                                }))}
                            />
                            <TextInput
                                label="End" type="number" style={{width:80}}
                                value={String(windowSpec.target.end_idx)}
                                onChange={e=>setWindowSpec(ws=>({
                                    ...ws,
                                    target:{...ws.target,end_idx:+e.currentTarget.value||0}
                                }))}
                            />
                        </Group>
                    </>
                )}

                {/* The rest of your controls: split, tuning, algorithm, preview, train... */}
                <Divider mb="sm" mt="sm"/>
                <Select
                    label="Train/Test Split Method"
                    data={[
                        { value: 'random', label: 'Random (shuffled)' },
                        { value: 'stratified', label: 'Stratified (for classification)' },
                        { value: 'time_series', label: 'Time Series (no shuffling, splits on order)' },
                        { value: 'kfold', label: 'K-Fold CV (advanced)' },
                    ]}
                    value={splitMethod}
                    onChange={v => setSplitMethod(v as SplitMethod)}
                    mb="md"
                />
                {splitMethod !== 'kfold' && (
                    <TextInput
                        label="Test Set Ratio"
                        type="number"
                        min={0.05}
                        max={0.5}
                        step={0.01}
                        value={String(splitRatio)}
                        onChange={e => setSplitRatio(+e.currentTarget.value)}
                        mb="md"
                    />
                )}
                {splitMethod === 'kfold' && (
                    <TextInput
                        label="Number of Folds (K)"
                        type="number"
                        min={2}
                        max={10}
                        value={String(params.kfold ?? 5)}
                        onChange={e => setParams(p=>({...p, kfold:+e.currentTarget.value}))}
                    />
                )}
                <Divider mt="sm" mb="sm"/>
                <Select
                    label="Algorithm"
                    data={algorithmOptions}
                    value={algorithm}
                    onChange={val=>setAlgorithm(val as Algorithm)}
                    mb="md"
                    disabled={!inferredTaskType}
                    placeholder={inferredTaskType ? "Select an algorithm" : "Select a valid target column"}
                />

                {/* Tuning strategy */}
                <Select
                    label="Tuning Strategy"
                    data={[
                        {value:'manual', label:'Manual'},
                        {value:'random', label:'Random Search'},
                        {value:'grid', label:'Grid Search'},
                        {value:'halving', label:'Successive Halving'},
                        {value:'evolutionary', label:'Evolutionary'},
                    ]}
                    value={tuning}
                    onChange={val=>setTuning(val as Tuning)}
                    mb="md"
                />

                {/* Per‐strategy inputs */}
                {tuning==='manual' && <Group mb="md">{renderHyperparams()}</Group>}

                {tuning==='random' && (
                    <Group mb="md">
                        <TextInput
                            label="Iterations"
                            type="number"
                            value={String(hpoIters)}
                            onChange={e=>setHpoIters(+e.currentTarget.value||0)}
                            style={{flex:1}}
                        />
                        <TextInput
                            label="CV Folds"
                            type="number"
                            value={String(hpoCv)}
                            onChange={e=>setHpoCv(+e.currentTarget.value||0)}
                            style={{flex:1}}
                        />
                    </Group>
                )}

                {tuning==='grid' && (
                    <Group mb="md">
                        <TextInput
                            label="CV Folds"
                            type="number"
                            value={String(hpoCv)}
                            onChange={e=>setHpoCv(+e.currentTarget.value||0)}
                            style={{width: '100%'}}
                        />
                    </Group>
                )}

                {tuning==='halving' && (
                    <Group mb="md">
                        <TextInput
                            label="CV Folds"
                            type="number"
                            value={String(hpoCv)}
                            onChange={e=>setHpoCv(+e.currentTarget.value||0)}
                            style={{flex:1}}
                        />
                        <TextInput
                            label="Factor"
                            type="number"
                            value={String(hpoFactor)}
                            onChange={e=>setHpoFactor(+e.currentTarget.value||0)}
                            style={{flex:1}}
                        />
                        <TextInput
                            label="Min Resources"
                            type="number"
                            value={String(hpoMinResources)}
                            onChange={e=>setHpoMinResources(+e.currentTarget.value||0)}
                            style={{flex:1}}
                        />
                    </Group>
                )}

                {tuning==='evolutionary' && (
                    <Group mb="md">
                        <TextInput
                            label="Population Size"
                            type="number"
                            value={String(hpoPop)}
                            onChange={e=>setHpoPop(+e.currentTarget.value||0)}
                            style={{flex:1}}
                        />
                        <TextInput
                            label="Generations"
                            type="number"
                            value={String(hpoGens)}
                            onChange={e=>setHpoGens(+e.currentTarget.value||0)}
                            style={{flex:1}}
                        />
                    </Group>
                )}

                {rayAlgos.includes(algorithm) && (
                    <Notification color="indigo" mt="xs" mb="md">
                        This algorithm uses Ray for distributed training. <br />
                        Make sure your Ray cluster is running and accessible from this backend.
                    </Notification>
                )}

                <Group mb="md">
                    <Button onClick={handlePreview} loading={previewing} disabled={disablePreview}>
                        Preview
                    </Button>
                    {previewOK    && <Notification color="teal" icon={<IconCheck/>}>Preview OK</Notification>}
                    {previewError && <Notification color="red"  icon={<IconX/>}>{previewError}</Notification>}
                </Group>

                <Group mt="xl">
                    <Button onClick={handleTrain} loading={loading} disabled={!previewOK}>
                        Train
                    </Button>
                    {error   && <Notification color="red" icon={<IconX/>}>{error}</Notification>}
                    {success && <Notification color="teal" icon={<IconCheck/>}>Queued!</Notification>}
                </Group>
            </Drawer>
        </>
    );
}