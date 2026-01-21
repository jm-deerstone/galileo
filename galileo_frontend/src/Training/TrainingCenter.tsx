// src/TrainingCenter.tsx
import React, { useState, useEffect, useMemo } from 'react';
import {
    Drawer,
    Tabs,
    Title,
    Text,
    Table,
    Flex,
    Select,
    Button,
    Group,
    Loader,
    Center,
    Notification,
    Badge,
    Box,
    Divider,
    MultiSelect,
    Switch,
    TextInput,
    Stack,
    SegmentedControl,
    Tooltip,
} from '@mantine/core';
import {
    IconPhoto,
    IconManualGearbox,
    IconAutomation,
    IconInfoCircle,
    IconTrash,
} from '@tabler/icons-react';
import { LineChart } from '@mantine/charts';
import { Progress } from '@mantine/core';

interface SnapshotInfo { id: string; created_at: string; }
interface InputColumnSchema { name: string; dtype: string; nullable?: boolean; }
interface InputSchema { columns: InputColumnSchema[]; }
interface SlidingWindowFeature { name: string; start_idx: number; end_idx: number; }

interface ConfigJSON {
    algorithm: string;
    params: Record<string, any>;
    window_spec: Record<string, any>;
    features?: string[];
    target?: string;
}

interface TrainingRead {
    id: string;
    name: string;
    datasource_id: string;
    config_json: ConfigJSON;
    input_schema_json: InputSchema;
}

interface ExecutionRead {
    id: string;
    snapshot_id: string;
    status: string;
    started_at: string;
    finished_at: string | null;
    metrics_json: Record<string, any>;
    model_path: string;
}

interface Props {
    id: string;
    opened: boolean;
    onClose(): void;
    onDelete(): void;
}

type ValidationCurveObj = {
    curve: { param_range: number[]; train_scores: number[]; valid_scores: number[]; [k: string]: any };
    param_range: number[];
    desc?: string;
    [k: string]: any;
};

const isObject = (v: any) => v && typeof v === 'object' && !Array.isArray(v);
function isNumberArray(arr: unknown): arr is number[] { return Array.isArray(arr) && arr.every(v => typeof v === 'number'); }

function getYDomain(data: any[], keys: string[]): [number, number] {
    const values = data.flatMap(d => keys.map(k => d[k])).filter(v => typeof v === 'number');
    if (!values.length) return [0, 1];
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) { min -= 1; max += 1; }
    const range = max - min;
    return [min - range * 0.05, max + range * 0.05];
}

interface CurveChartProps { name: string; curve: any; label?: string; secondary?: number[]; desc?: string; }
export function CurveChart({ name, curve, label, secondary, desc }: CurveChartProps) {
    if (isObject(curve) && isNumberArray(curve.param_range) && isNumberArray(curve.train_scores) && isNumberArray(curve.valid_scores)) {
        type DP = { param: number; train: number; val: number };
        const data: DP[] = curve.param_range.map((param: number, i: number) => ({ param, train: curve.train_scores[i], val: curve.valid_scores[i] }));
        const yDomain = getYDomain(data, ['train', 'val']);
        return (
            <Box my="xs">
                <Text size="sm" fw={500} mb={2}>{label || name}</Text>
                {desc && <Text c="dimmed" size="xs" mb={4}>{desc}</Text>}
                <LineChart
                    h={140}
                    data={data}
                    dataKey="param"
                    series={[{ name: 'train', color: 'red', label: 'Train' }, { name: 'val', color: 'blue', label: 'Validation' }]}
                    xAxisProps={{
                        label: 'Parameter Value',
                        tickFormatter: (v: number) => (Math.abs(v) < 1e-2 || Math.abs(v) > 1e2 ? v.toExponential(2) : v.toFixed(3)),
                    }}
                    yAxisProps={{ tickFormatter: (n: number) => n.toFixed(2), domain: yDomain }}
                    legendProps={{ align: 'center', verticalAlign: 'top' }}
                />
            </Box>
        );
    }

    if (isObject(curve) && isNumberArray(curve.train_sizes) && isNumberArray(curve.train_scores) && isNumberArray(curve.valid_scores)) {
        type DP = { size: number; train: number; val: number };
        const data: DP[] = curve.train_sizes.map((size: number, i: number) => ({ size, train: curve.train_scores[i], val: curve.valid_scores[i] }));
        const yDomain = getYDomain(data, ['train', 'val']);
        return (
            <Box my="xs">
                <Text size="sm" fw={500} mb={2}>{label || name}</Text>
                {desc && <Text c="dimmed" size="xs" mb={4}>{desc}</Text>}
                <LineChart
                    h={140}
                    data={data}
                    dataKey="size"
                    series={[{ name: 'train', color: 'red', label: 'Train' }, { name: 'val', color: 'blue', label: 'Validation' }]}
                    xAxisProps={{ label: 'Training Size' }}
                    yAxisProps={{ tickFormatter: (n: number) => n.toFixed(2), domain: yDomain }}
                    legendProps={{ align: 'center', verticalAlign: 'top' }}
                />
            </Box>
        );
    }

    if (isObject(curve) && isNumberArray(curve.train_scores) && isNumberArray(curve.valid_scores) && !curve.param_range && !curve.train_sizes) {
        type DP = { step: number; train: number; val: number };
        const data: DP[] = curve.train_scores.map((v: number, i: number) => ({ step: i + 1, train: v, val: curve.valid_scores[i] }));
        const yDomain = getYDomain(data, ['train', 'val']);
        return (
            <Box my="xs">
                <Text size="sm" fw={500} mb={2}>{label || name}</Text>
                {desc && <Text c="dimmed" size="xs" mb={4}>{desc}</Text>}
                <LineChart
                    h={140}
                    data={data}
                    dataKey="step"
                    series={[{ name: 'train', color: 'red', label: 'Train' }, { name: 'val', color: 'blue', label: 'Validation' }]}
                    xAxisProps={{ label: 'Step' }}
                    yAxisProps={{ tickFormatter: (n: number) => n.toFixed(2), domain: yDomain }}
                    legendProps={{ align: 'center', verticalAlign: 'top' }}
                />
            </Box>
        );
    }

    if (isNumberArray(curve)) {
        type DP = Record<string, number>;
        const data: DP[] = curve.map((v: number, i: number) => ({ step: i + 1, [name]: v, ...(secondary?.length ? { [`${name}_val`]: secondary[i] } : {}) }));
        const keys = [name, ...(secondary?.length ? [`${name}_val`] : [])];
        const yDomain = getYDomain(data, keys);
        return (
            <Box my="xs">
                <Text size="sm" fw={500} mb={2}>{label || name}</Text>
                {desc && <Text c="dimmed" size="xs" mb={4}>{desc}</Text>}
                <LineChart
                    h={140}
                    data={data}
                    dataKey="step"
                    series={[
                        { name, color: 'red', label: 'Train' },
                        ...(secondary?.length ? [{ name: `${name}_val`, color: 'blue', label: 'Validation' }] : []),
                    ]}
                    curveType="linear"
                    gridAxis="xy"
                    yAxisProps={{ tickFormatter: (n: number) => n.toFixed(2), domain: yDomain }}
                    xAxisProps={{ label: 'Step' }}
                    legendProps={{ align: 'center', verticalAlign: 'top' }}
                />
            </Box>
        );
    }

    if (Array.isArray(curve) || isObject(curve)) {
        return (
            <Box my="xs">
                <Text size="sm" fw={500} mb={2}>{label || name}</Text>
                {desc && <Text c="dimmed" size="xs" mb={4}>{desc}</Text>}
                <pre style={{ fontSize: 12, background: '#1113', borderRadius: 4, padding: 4, overflowX: 'auto' }}>
          {JSON.stringify(curve, null, 2)}
        </pre>
            </Box>
        );
    }
    return null;
}

export const TrainingCenter: React.FC<Props> = ({ id, opened, onClose, onDelete }) => {
    const [tr, setTr] = useState<TrainingRead | null>(null);
    const [executions, setExecutions] = useState<ExecutionRead[]>([]);
    const [snapshots, setSnapshots] = useState<SnapshotInfo[]>([]);
    const [selectedSnap, setSelectedSnap] = useState<string | null>(null);
    const [running, setRunning] = useState(false);
    const [runError, setRunError] = useState<string | null>(null);
    const [runSuccess, setRunSuccess] = useState<string | null>(null);

    const [metricsDrawerOpen, setMetricsDrawerOpen] = useState(false);
    const [metricsDrawerExec, setMetricsDrawerExec] = useState<ExecutionRead | null>(null);
    const [progressMap, setProgressMap] = useState<Record<string, any>>({});

    useEffect(() => {
        const runningExecs = executions.filter(e => e.status === 'running');
        if (!runningExecs.length) return;

        const fetchProgress = async () => {
            for (const exec of runningExecs) {
                try {
                    const resp = await fetch(`http://localhost:8000/training_executions/${exec.id}/progress/`);
                    if (!resp.ok) continue;
                    const data = await resp.json();
                    setProgressMap(pm => ({ ...pm, [exec.id]: data }));
                } catch { /* ignore */ }
            }
        };

        fetchProgress();
        const interval = setInterval(fetchProgress, 2000);
        return () => clearInterval(interval);
    }, [executions]);

    // ---------- Automation ----------
    const [autoConfig, setAutoConfig] = useState({
        automation_enabled: false,
        automation_schedule: '',
        promotion_metrics: [] as string[],
    });
    const [autoExists, setAutoExists] = useState<boolean>(false); // <- true if GET 200
    const [autoLoading, setAutoLoading] = useState(false);
    const [autoError, setAutoError] = useState<string | null>(null);
    const [autoSuccess, setAutoSuccess] = useState<string | null>(null);
    const [autoRunOK, setAutoRunOK] = useState<string | null>(null);
    const [autoRunErr, setAutoRunErr] = useState<string | null>(null);
    const [autoRunning, setAutoRunning] = useState(false);
    // Schedule UI — align with AutomationDrawer (seconds or cron, no prefixes)
    type ScheduleMode = 'interval' | 'cron';
    const [mode, setMode] = useState<ScheduleMode>('interval');
    const [intervalSec, setIntervalSec] = useState<string>('900');          // default 15 min
    const [cron, setCron] = useState<string>('*/15 * * * *');               // default every 15m

    // Derive the outgoing value we'll PUT to the backend
    const scheduleOut = useMemo(() => {
        if (!autoConfig.automation_enabled) return null;
        if (mode === 'interval') {
            const s = (intervalSec || '').trim();
            return /^\d+$/.test(s) ? s : null; // numeric seconds only
        }
        const c = (cron || '').trim();
        return c.length ? c : null;          // standard crontab string
    }, [autoConfig.automation_enabled, mode, intervalSec, cron]);
    const [promoting, setPromoting] = useState<string | null>(null);
    const [deployments, setDeployments] = useState<any[]>([]);

    const METRIC_LABELS: Record<string, string> = { accuracy: 'Accuracy', r2: 'R²', mse: 'MSE', rmse: 'RMSE' };

    useEffect(() => {
        if (!tr) return;
        fetch(`http://localhost:8000/deployments/by_training/${tr.id}/`).then(r => r.json()).then(setDeployments);
    }, [tr]);

    function getDeploymentStatuses(execId: string): string[] {
        return deployments.filter(d => d.training_execution_id === execId)
            .map(d => d.metric ? d.metric : (d.promotion_type === 'manual' ? 'Manual' : '—'));
    }

    const [allowedMetrics, setAllowedMetrics] = useState<string[]>([]);
    useEffect(() => {
        if (!opened) return;
        (async () => {
            // Load automation config; 404 => no automation created yet
            try {
                setAutoLoading(true);
                const resp = await fetch(`http://localhost:8000/trainings/${id}/automation_config/`);
                if (resp.status === 404) {
                    setAutoExists(false);
                    setAutoConfig({ automation_enabled: false, automation_schedule: '', promotion_metrics: [] });
                } else if (!resp.ok) {
                    throw new Error('Failed to load automation config');
                } else {
                    const data = await resp.json();
                    setAutoExists(true);
                    setAutoConfig({
                        automation_enabled: !!data.automation_enabled,
                        automation_schedule: data.automation_schedule || '',
                        promotion_metrics: Array.isArray(data.promotion_metrics) ? data.promotion_metrics : [],
                    });
                    if (data?.automation_schedule && /^\d+$/.test(String(data.automation_schedule))) {
                        setMode('interval');
                        setIntervalSec(String(data.automation_schedule));
                        setCron('*/15 * * * *');
                    } else {
                        setMode('cron');
                        setCron(data?.automation_schedule || '*/15 * * * *');
                        setIntervalSec('900');
                    }
                }
            } catch (err: any) {
                setAutoError(err.message);
            } finally {
                setAutoLoading(false);
            }

            // Load allowed metrics
            try {
                const resp = await fetch(`http://localhost:8000/trainings/${id}/allowed_metrics/`);
                if (resp.ok) setAllowedMetrics(await resp.json());
                else setAllowedMetrics([]);
            } catch {
                setAllowedMetrics([]);
            }
        })();
    }, [opened, id]);

    // Schedule UI
    const intervalOptions = [
        { value: 'interval:1h', label: 'Every hour' },
        { value: 'interval:6h', label: 'Every 6 hours' },
        { value: 'interval:12h', label: 'Every 12 hours' },
        { value: 'interval:1d', label: 'Every day' },
        { value: 'custom', label: 'Custom interval…' },
    ];

    const getScheduleType = (s: string) => s?.startsWith('cron:') ? 'cron' : s?.startsWith('interval:') ? 'interval' : '';
    const getScheduleValue = (s: string) => s?.replace(/^(cron:|interval:)/, '') || '';

    const [scheduleType, setScheduleType] = useState<'interval' | 'cron' | ''>('');
    const [intervalValue, setIntervalValue] = useState<string>('interval:1d');
    const [customInterval, setCustomInterval] = useState<string>('');
    const [cronValue, setCronValue] = useState<string>('');

    useEffect(() => {
        const sType = getScheduleType(autoConfig.automation_schedule || '');
        setScheduleType((sType as any) || 'interval');

        if (sType === 'interval') {
            const raw = autoConfig.automation_schedule || 'interval:1d';
            const isPreset = intervalOptions.some(opt => opt.value === raw);
            if (isPreset) { setIntervalValue(raw); setCustomInterval(''); }
            else { setIntervalValue('custom'); setCustomInterval(getScheduleValue(raw)); }
            setCronValue('');
        } else if (sType === 'cron') {
            setCronValue(getScheduleValue(autoConfig.automation_schedule || ''));
        } else {
            setIntervalValue('interval:1d'); setCronValue(''); setCustomInterval('');
        }
    }, [autoConfig.automation_schedule]);

    const handleScheduleChange = (type: string, value?: string) => {
        setScheduleType(type as any);
        if (type === 'interval') {
            if (value && value !== 'custom') {
                setIntervalValue(value);
                setAutoConfig(ac => ({ ...ac, automation_schedule: value }));
            } else if (value === 'custom') {
                setIntervalValue('custom');
                setAutoConfig(ac => ({ ...ac, automation_schedule: `interval:${customInterval}` }));
            }
        } else if (type === 'cron') {
            setAutoConfig(ac => ({ ...ac, automation_schedule: `cron:${cronValue}` }));
        }
    };

    const metricOptions = useMemo(() => {
        // Union allowed + currently configured, so existing values always render in the widget
        const union = Array.from(new Set([...(allowedMetrics || []), ...(autoConfig.promotion_metrics || [])]));
        return union.map(m => ({ value: m, label: METRIC_LABELS[m] || m }));
    }, [allowedMetrics, autoConfig.promotion_metrics]);

    const saveAutomationConfig = async () => {
        setAutoLoading(true);
        setAutoError(null);
        setAutoSuccess(null);
        try {
            const resp = await fetch(
                `http://localhost:8000/trainings/${id}/automation_config/`,
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        automation_enabled: autoConfig.automation_enabled,
                        automation_schedule: scheduleOut,              // <— seconds or cron, or null
                        promotion_metrics: autoConfig.promotion_metrics,
                    }),
                }
            );
            if (!resp.ok) throw new Error('Failed to save automation config');
            setAutoExists(true);
            setAutoSuccess('Automation settings saved!');
        } catch (err: any) {
            setAutoError(err.message);
        } finally {
            setAutoLoading(false);
        }
    };


    const deleteAutomation = async () => {
        if (!window.confirm('Delete automation for this training? This cannot be undone.')) return;
        setAutoLoading(true);
        setAutoError(null);
        setAutoSuccess(null);
        try {
            const resp = await fetch(`http://localhost:8000/trainings/${id}/automation_config/`, { method: 'DELETE' });
            if (!(resp.ok || resp.status === 204)) throw new Error('Failed to delete automation');
            setAutoExists(false);
            setAutoConfig({ automation_enabled: false, automation_schedule: '', promotion_metrics: [] });

            setAutoSuccess('Automation deleted.');
        } catch (err: any) {
            setAutoError(err.message);
        } finally {
            setAutoLoading(false);
        }
    };

    const runAutomationNow = async () => {
        setAutoRunning(true);
        setAutoRunOK(null);
        setAutoRunErr(null);
        try {
            const resp = await fetch(`http://localhost:8000/trainings/${id}/automation/run_now`, { method: 'POST' });
            if (!resp.ok) {
                const j = await resp.json().catch(() => ({}));
                throw new Error(j.detail || `Run now failed (HTTP ${resp.status})`);
            }
            setAutoRunOK('Pipeline triggered — check training executions for progress.');
        } catch (e: any) {
            setAutoRunErr(e.message);
        } finally {
            setAutoRunning(false);
        }
    };

    const [currentDeployment, setCurrentDeployment] = useState<any>(null);
    useEffect(() => {
        if (!tr) return;
        (async () => {
            try {
                const depResp = await fetch(`http://localhost:8000/deployments/?training_id=${tr.id}`);
                if (!depResp.ok) return;
                const dps = await depResp.json();
                if (dps && dps.length > 0) setCurrentDeployment(dps[0]);
            } catch {}
        })();
    }, [tr]);

    useEffect(() => {
        const runningExecs = executions.filter(e => e.status === 'running');
        if (!runningExecs.length) return;
        let interval: ReturnType<typeof setInterval> | null = null;

        const fetchProgress = async () => {
            let anyActive = false;
            for (const exec of runningExecs) {
                try {
                    const resp = await fetch(`http://localhost:8000/training_executions/${exec.id}/progress/`);
                    if (!resp.ok) continue;
                    const data = await resp.json();
                    setProgressMap(pm => ({ ...pm, [exec.id]: data }));
                    if (!(data.stage === 'done' || data.stage === 'refit' || data.status === 'done' || (typeof data.progress === 'number' && data.progress >= 1))) {
                        anyActive = true;
                    }
                } catch { /* ignore */ }
            }
            if (!anyActive && interval) clearInterval(interval);
        };

        fetchProgress();
        interval = setInterval(fetchProgress, 2000);
        return () => { if (interval) clearInterval(interval); };
    }, [executions]);

    const promoteManual = async (execId: string) => {
        setPromoting(execId);
        try {
            const resp = await fetch(`http://localhost:8000/trainings/${id}/executions/${execId}/promote_manual/`, { method: 'POST' });
            if (!resp.ok) throw new Error('Failed to promote manually');
            setRunSuccess('Model promoted manually and locked.');
        } catch (e: any) {
            setRunError(e.message);
        } finally {
            setPromoting(null);
        }
    };

    function getProgressValue(p?: any) {
        if (!p) return 0;
        if (typeof p.progress === 'number') return p.progress > 1 ? p.progress : p.progress * 100;
        if (typeof p.generation === 'number' && typeof p.total_generations === 'number' && p.total_generations > 0) {
            return (p.generation / p.total_generations) * 100;
        }
        return 0;
    }

    useEffect(() => {
        if (!opened) return;
        (async () => {
            setTr(null);
            setSnapshots([]);
            setSelectedSnap(null);
            try {
                const trResp = await fetch(`http://localhost:8000/trainings/${id}/`);
                if (!trResp.ok) throw new Error(`Training ${trResp.status}`);
                const trData: TrainingRead = await trResp.json();
                setTr(trData);

                const exResp = await fetch(`http://localhost:8000/trainings/${id}/executions/`);
                if (!exResp.ok) throw new Error(`Executions ${exResp.status}`);
                const rawEx: any[] = await exResp.json();
                const parsedEx: ExecutionRead[] = rawEx.map(e => {
                    let metrics: any = {};
                    if (typeof e.metrics_json === 'string' && e.metrics_json.trim()) {
                        try { metrics = JSON.parse(e.metrics_json); } catch { metrics = { _raw: e.metrics_json }; }
                    } else { metrics = e.metrics_json || {}; }
                    return { ...e, metrics_json: metrics };
                });
                setExecutions(parsedEx);

                const dsResp = await fetch(`http://localhost:8000/datasources/${trData.datasource_id}/with-snapshot/`);
                if (!dsResp.ok) throw new Error(`Snapshots ${dsResp.status}`);
                const dsData = await dsResp.json();
                setSnapshots(dsData.snapshots);
            } catch (err) { console.error(err); }
        })();
    }, [opened, id]);

    const handleRun = async () => {
        if (!tr || !selectedSnap) { setRunError('Please select a snapshot'); return; }
        setRunning(true); setRunError(null); setRunSuccess(null);
        try {
            const resp = await fetch(`http://localhost:8000/trainings/${id}/execute/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ snapshot_id: selectedSnap }),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Execution failed');
            }
            setRunSuccess('Training queued');
            const exData = await fetch(`http://localhost:8000/trainings/${id}/executions/`).then(r => r.json());
            setExecutions(exData as ExecutionRead[]);
        } catch (err: any) {
            setRunError(err.message);
        } finally {
            setRunning(false);
        }
    };

    const renderFeatures = () => {
        if (tr!.config_json.features) return tr!.config_json.features.join(', ');
        const ws = tr!.config_json.window_spec;
        if (!ws?.features?.length) return '—';
        return ws.features.map((f: SlidingWindowFeature) => `${f.name}[${f.start_idx}→${f.end_idx}]`).join(', ');
    };
    const renderTarget = () => {
        if (tr!.config_json.target) return tr!.config_json.target;
        const tw = tr!.config_json.window_spec?.target as SlidingWindowFeature | undefined;
        if (!tw?.name) return '—';
        return `${tw.name}[${tw.start_idx}→${tw.end_idx}]`;
    };

    return (
        <Drawer opened={opened} onClose={onClose} title="Training Center" position="bottom" size="80%">
            {!tr ? (
                <Center style={{ height: 200 }}><Loader /></Center>
            ) : (
                <Tabs defaultValue="executions">
                    <MetricsDrawer opened={metricsDrawerOpen} onClose={() => setMetricsDrawerOpen(false)} execution={metricsDrawerExec} />

                    <Tabs.List>
                        <Tabs.Tab value="executions" leftSection={<IconPhoto size={14} />}>History & Run</Tabs.Tab>
                        <Tabs.Tab value="automation" leftSection={<IconAutomation />}>Automation</Tabs.Tab>
                    </Tabs.List>

                    {/* --------- EXECUTIONS --------- */}
                    <Tabs.Panel value="executions" pt="md">
                        <Flex justify="space-between" align="center">
                            <Title order={4}>Training Configuration</Title>
                            <Button
                                variant="outline"
                                color="red"
                                onClick={async () => {
                                    if (!window.confirm('Really delete this training and all executions?')) return;
                                    await fetch(`http://localhost:8000/trainings/${id}/delete`, { method: 'DELETE' });
                                    onDelete();
                                    onClose();
                                }}
                            >
                                Delete Training
                            </Button>
                        </Flex>

                        <Table striped mb="md">
                            <Table.Thead>
                                <Table.Tr>
                                    <Table.Th>Algorithm</Table.Th>
                                    <Table.Th>Params</Table.Th>
                                    <Table.Th>Features / Windows</Table.Th>
                                    <Table.Th>Target / Window</Table.Th>
                                    {Object.keys(tr.config_json)
                                        .filter(k => !['algorithm', 'params', 'features', 'target', 'window_spec'].includes(k))
                                        .map(k => (<Table.Th key={k}>{k.replace(/_/g, ' ')}</Table.Th>))}
                                </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                                <Table.Tr>
                                    <Table.Td><Text>{tr.config_json.algorithm}, {tr.id}</Text></Table.Td>
                                    <Table.Td><pre style={{ margin: 0, fontSize: '0.85em' }}>{JSON.stringify(tr.config_json.params, null, 2)}</pre></Table.Td>
                                    <Table.Td><Text>{renderFeatures()}</Text></Table.Td>
                                    <Table.Td><Text>{renderTarget()}</Text></Table.Td>
                                    {Object.entries(tr.config_json)
                                        .filter(([k]) => !['algorithm', 'params', 'features', 'target', 'window_spec'].includes(k))
                                        .map(([k, v]) => (<Table.Td key={k}>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</Table.Td>))}
                                </Table.Tr>
                            </Table.Tbody>
                        </Table>

                        <Divider mb="md" />
                        <Title order={4}>Execution History</Title>
                        <Table striped mb="md">
                            <Table.Thead>
                                <Table.Tr>
                                    <Table.Th>ID</Table.Th>
                                    <Table.Th>Input Snapshot</Table.Th>
                                    <Table.Th>Status</Table.Th>
                                    <Table.Th>Started</Table.Th>
                                    <Table.Th>Finished</Table.Th>
                                    <Table.Th>Metrics</Table.Th>
                                    <Table.Th>Deployment Status</Table.Th>
                                    <Table.Th>Actions</Table.Th>
                                </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                                {executions.map(e => (
                                    <Table.Tr key={e.id}>
                                        <Table.Td>{e.id.slice(0, 8)}…</Table.Td>
                                        <Table.Td>{e.snapshot_id.slice(0, 8)}…</Table.Td>
                                        <Table.Td>
                                            <Flex direction="row" gap="15px">
                                                <Badge color={e.status === 'success' ? 'teal' : e.status === 'running' ? 'blue' : 'red'}>{e.status}</Badge>
                                                {e.status === 'running' && (
                                                    <Box mb="sm">
                                                        <Progress value={getProgressValue(progressMap[e.id])} striped animated size="md" />
                                                        <pre style={{ fontSize: 10, margin: 0 }}>{JSON.stringify(progressMap[e.id], null, 2)}</pre>
                                                    </Box>
                                                )}
                                            </Flex>
                                        </Table.Td>
                                        <Table.Td>{new Date(e.started_at).toLocaleString()}</Table.Td>
                                        <Table.Td>{e.finished_at ? new Date(e.finished_at).toLocaleString() : '–'}</Table.Td>
                                        <Table.Td>
                                            <Button size="xs" variant="light" onClick={() => { setMetricsDrawerExec(e); setMetricsDrawerOpen(true); }}>
                                                Show Metrics
                                            </Button>
                                        </Table.Td>
                                        <Table.Td>
                                            {getDeploymentStatuses(e.id).length
                                                ? getDeploymentStatuses(e.id).map(status => <Badge key={status} color={status === 'Manual' ? 'gray' : 'teal'}>{status}</Badge>)
                                                : <Text c="dimmed" size="xs">–</Text>}
                                        </Table.Td>
                                        <Table.Td>
                                            <Button size="xs" variant="light" loading={promoting === e.id} onClick={() => promoteManual(e.id)} disabled={e.status !== 'success'}>
                                                Promote to Production (Manual)
                                            </Button>
                                        </Table.Td>
                                    </Table.Tr>
                                ))}
                            </Table.Tbody>
                        </Table>

                        <Divider mb="md" />
                        <Title order={4}>Run Training</Title>
                        <Group mb="md" align="flex-end">
                            <Select
                                style={{ flex: 1 }}
                                label="Select Snapshot"
                                placeholder="Pick a snapshot"
                                data={snapshots.map(s => ({ value: s.id, label: new Date(s.created_at).toLocaleString() }))}
                                value={selectedSnap}
                                onChange={setSelectedSnap}
                            />
                            <Button leftSection={<IconManualGearbox size={16} />} onClick={handleRun} loading={running} disabled={!selectedSnap}>
                                Execute
                            </Button>
                        </Group>

                        {runError && <Notification color="red" onClose={() => setRunError(null)} mb="sm">{runError}</Notification>}
                        {runSuccess && <Notification color="teal" onClose={() => setRunSuccess(null)} mb="sm">{runSuccess}</Notification>}
                    </Tabs.Panel>

                    {/* --------- AUTOMATION --------- */}
                    <Tabs.Panel value="automation" pt="md">
                        <Title order={4} mb="md">Automation Settings</Title>
                        <Divider mb="sm" />

                        {!autoExists ? (
                            <Box maw={520} mx="auto">
                                <Notification color="gray" withCloseButton={false} mb="md">
                                    No automation is configured for this training yet.
                                    <br />
                                    Use the <b>Automation</b> button in the left sidebar to create a schedule for this training.
                                </Notification>
                            </Box>
                        ) : (
                            <>
                                {currentDeployment && (
                                    <Group mb="md">
                                        <Badge color={currentDeployment.locked ? 'gray' : 'teal'}>
                                            {currentDeployment.locked ? 'Manual (Locked)' : 'Auto-managed'}
                                        </Badge>
                                        <Button
                                            size="xs"
                                            variant={currentDeployment.locked ? 'outline' : 'light'}
                                            onClick={async () => {
                                                await fetch(`http://localhost:8000/model_deployments/${currentDeployment.id}/lock/?lock=${!currentDeployment.locked}`, { method: 'POST' });
                                            }}
                                        >
                                            {currentDeployment.locked ? 'Unlock for Automation' : 'Lock Deployment'}
                                        </Button>
                                    </Group>
                                )}

                                <Box maw={520} mx="auto">
                                    <Stack gap="sm">
                                        <Switch
                                            label="Enable Automation"
                                            checked={autoConfig.automation_enabled}
                                            onChange={(e) =>
                                                setAutoConfig((ac) => ({ ...ac, automation_enabled: e.currentTarget.checked }))
                                            }
                                        />

                                        <SegmentedControl
                                            value={mode}
                                            onChange={(v) => setMode(v as ScheduleMode)}
                                            data={[
                                                { label: 'Interval (sec)', value: 'interval' },
                                                { label: 'Cron', value: 'cron' },
                                            ]}
                                            disabled={!autoConfig.automation_enabled}
                                        />

                                        {mode === 'interval' ? (
                                            <TextInput
                                                label="Interval seconds"
                                                description="Number of seconds between runs (e.g., 900 for 15 minutes)."
                                                value={intervalSec}
                                                onChange={(e) => setIntervalSec(e.currentTarget.value)}
                                                disabled={!autoConfig.automation_enabled}
                                            />
                                        ) : (
                                            <TextInput
                                                label="Cron schedule"
                                                description="Standard crontab (e.g., */15 * * * *)."
                                                value={cron}
                                                onChange={(e) => setCron(e.currentTarget.value)}
                                                disabled={!autoConfig.automation_enabled}
                                            />
                                        )}

                                        <MultiSelect
                                            label="Promotion metrics (optional)"
                                            description="If set, the system will automatically keep the best execution per metric."
                                            data={(allowedMetrics || []).map((m) => ({ value: m, label: m }))}
                                            value={autoConfig.promotion_metrics}
                                            onChange={(vals) =>
                                                setAutoConfig((ac) => ({ ...ac, promotion_metrics: vals }))
                                            }
                                            searchable
                                            nothingFoundMessage="No metrics"
                                        />

                                        <Group mt="md">
                                            <Button onClick={saveAutomationConfig} loading={autoLoading}>
                                                Save Automation Settings
                                            </Button>
                                            <Button
                                                variant="light"
                                                onClick={runAutomationNow}
                                                loading={autoRunning}
                                                disabled={!autoConfig.automation_enabled}
                                            >
                                                Run Now
                                            </Button>
                                            <Button
                                                variant="outline"
                                                color="red"
                                                leftSection={<IconTrash size={16} />}
                                                onClick={deleteAutomation}
                                            >
                                                Delete Automation
                                            </Button>
                                        </Group>

                                        {/* (Optional) show what will be saved */}
                                        <Box mt="xs">
                                            <Text size="xs" c="dimmed" mb={4}>Payload preview:</Text>
                                            <pre style={{ fontSize: 12, background: '#1113', borderRadius: 4, padding: 8, overflowX: 'auto' }}>
                                                {JSON.stringify({
                                                    automation_enabled: autoConfig.automation_enabled,
                                                    automation_schedule: scheduleOut,
                                                    promotion_metrics: autoConfig.promotion_metrics
                                                }, null, 2)}
                                            </pre>
                                        </Box>
                                    </Stack>

                                </Box>

                                <Box mt="md" maw={520} mx="auto">
                                    {autoError && <Notification color="red" onClose={() => setAutoError(null)} mb="sm">{autoError}</Notification>}
                                    {autoSuccess && <Notification color="teal" onClose={() => setAutoSuccess(null)} mb="sm">{autoSuccess}</Notification>}
                                    {autoRunErr && <Notification color="red" onClose={() => setAutoRunErr(null)} mb="sm">{autoRunErr}</Notification>}
                                    {autoRunOK && <Notification color="indigo" onClose={() => setAutoRunOK(null)} mb="sm">{autoRunOK}</Notification>}
                                </Box>
                            </>
                        )}
                    </Tabs.Panel>
                </Tabs>
            )}
        </Drawer>
    );
};

export function MetricsDrawer({
                                  opened,
                                  onClose,
                                  execution,
                              }: { opened: boolean; onClose: () => void; execution: any; }) {
    if (!execution) return null;
    const e = execution;
    return (
        <Drawer opened={opened} onClose={onClose} title="Metrics" size="xl" position="right">
            <Title order={4} mb="xs">Metrics for Execution {e.id.slice(0, 8)}…</Title>
            <Box>
                {typeof e.metrics_json.train_time_s === 'number' && (<Text><strong>Training Time:</strong> {e.metrics_json.train_time_s}s</Text>)}
                {typeof e.metrics_json.error === 'string' && (<Text c="red"><strong>Error:</strong> {e.metrics_json.error}</Text>)}
                <Divider my="sm" />
                <Table withColumnBorders withRowBorders striped>
                    <Table.Thead>
                        <Table.Tr><Table.Th>Metric</Table.Th><Table.Th>Train</Table.Th><Table.Th>Test</Table.Th></Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                        <Table.Tr><Table.Td><strong>Samples</strong></Table.Td><Table.Td>{e.metrics_json.n_samples_train ?? '—'}</Table.Td><Table.Td>{e.metrics_json.n_samples_test ?? '—'}</Table.Td></Table.Tr>
                        <Table.Tr><Table.Td><strong>MSE</strong></Table.Td><Table.Td>{e.metrics_json.mse?.toFixed?.(3) ?? '—'}</Table.Td><Table.Td>{e.metrics_json.test_mse?.toFixed?.(3) ?? '—'}</Table.Td></Table.Tr>
                        <Table.Tr><Table.Td><strong>RMSE</strong></Table.Td><Table.Td>{e.metrics_json.rmse?.toFixed?.(3) ?? '—'}</Table.Td><Table.Td>{e.metrics_json.test_rmse?.toFixed?.(3) ?? '—'}</Table.Td></Table.Tr>
                        <Table.Tr><Table.Td><strong>R²</strong></Table.Td><Table.Td>{e.metrics_json.r2?.toFixed?.(3) ?? '—'}</Table.Td><Table.Td>{e.metrics_json.test_r2?.toFixed?.(3) ?? '—'}</Table.Td></Table.Tr>
                    </Table.Tbody>
                </Table>
                <Divider my="sm" />
                {(() => {
                    const validationCurves = (e.metrics_json.validation_curves && typeof e.metrics_json.validation_curves === 'object')
                        ? e.metrics_json.validation_curves : null;
                    const curvesToRender: React.ReactNode[] = [];

                    if (validationCurves) {
                        Object.entries(validationCurves).forEach(([paramName, curveObjRaw]) => {
                            const curveObj = curveObjRaw as Partial<ValidationCurveObj> | null | undefined;
                            if (curveObj && typeof curveObj === 'object' && curveObj.curve && Array.isArray(curveObj.param_range)
                                && curveObj.curve && Array.isArray(curveObj.curve.train_scores) && Array.isArray(curveObj.curve.valid_scores)) {
                                curvesToRender.push(
                                    <CurveChart key={`validation_curve_${paramName}`} name={`validation_curve_${paramName}`} curve={curveObj.curve} label={`Validation Curve (${paramName})`} desc={curveObj.desc || ''} />
                                );
                            }
                        });
                    }

                    Object.entries(e.metrics_json)
                        .filter(([k, v]) => (k.endsWith('_curve') || k.endsWith('_curve_') || k === 'learning_curve' || k === 'validation_curve')
                            && (Array.isArray(v) || (v && typeof v === 'object')) && k !== 'validation_curves')
                        .forEach(([k, v]) => {
                            const baseName = k.replace(/^val_/, '').replace(/^train_/, '');
                            const train = e.metrics_json[`train_${baseName}`];
                            const val = e.metrics_json[`val_${baseName}`];
                            const desc = e.metrics_json[`${k}_desc`] || e.metrics_json[`${baseName}_desc`] || '';
                            if (k === 'learning_curve') {
                                curvesToRender.push(<CurveChart key={k} name={k} curve={v} label="Learning Curve" desc={desc} />);
                            } else {
                                curvesToRender.push(
                                    <CurveChart key={k} name={k} curve={train || v} secondary={val} label={baseName.replace(/_/g, ' ').replace(/\b\w/g, s => s.toUpperCase())} desc={desc} />
                                );
                            }
                        });

                    return curvesToRender.length ? curvesToRender : null;
                })()}
            </Box>
        </Drawer>
    );
}
