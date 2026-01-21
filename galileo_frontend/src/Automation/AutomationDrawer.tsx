import React, { useEffect, useMemo, useState } from 'react';
import {
    Badge,
    Button,
    Divider,
    Drawer,
    Group,
    MultiSelect,
    Notification,
    SegmentedControl,
    Select,
    Switch,
    Text,
    TextInput,
    Title,
} from '@mantine/core';
import { IconCheck, IconPlaystationX, IconRepeat, IconRocket } from '@tabler/icons-react';

type TrainingLite = { id: string; name: string; datasource_id: string };
type AutomationConfig = {
    automation_enabled: boolean;
    automation_schedule: string | null;
    promotion_metrics: string[];
};

type Props = {
    opened: boolean;
    onClose: () => void;
    /** Optional: when saved or run, we can notify the parent to refresh the graph */
    onChanged?: () => void;
};

type ScheduleMode = 'interval' | 'cron';

const API = 'http://localhost:8000';

export default function AutomationDrawer({ opened, onClose, onChanged }: Props) {
    const [trainings, setTrainings] = useState<TrainingLite[]>([]);
    const [selectedTrId, setSelectedTrId] = useState<string | null>(null);

    const [cfg, setCfg] = useState<AutomationConfig>({
        automation_enabled: false,
        automation_schedule: null,
        promotion_metrics: [],
    });

    const [allowedMetrics, setAllowedMetrics] = useState<string[]>([]);
    const [mode, setMode] = useState<ScheduleMode>('interval');
    const [intervalSec, setIntervalSec] = useState<string>('900'); // default 15 min
    const [cron, setCron] = useState<string>('*/15 * * * *');

    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [running, setRunning] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [savedOK, setSavedOK] = useState(false);
    const [runOK, setRunOK] = useState(false);

    // load trainings when drawer opens
    useEffect(() => {
        if (!opened) return;
        setError(null);
        fetch(`${API}/trainings/`)
            .then((r) => r.json())
            .then((arr: any[]) => {
                const list = arr.map((t) => ({ id: t.id, name: t.name, datasource_id: t.datasource_id })) as TrainingLite[];
                setTrainings(list);
                if (!selectedTrId && list.length) setSelectedTrId(list[0].id);
            })
            .catch((e) => setError(String(e)));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [opened]);

    // load automation config + allowed metrics for selected training
    useEffect(() => {
        if (!selectedTrId) return;
        setLoading(true);
        setError(null);
        Promise.all([
            fetch(`${API}/trainings/${selectedTrId}/automation_config/`).then((r) => r.json()),
            fetch(`${API}/trainings/${selectedTrId}/allowed_metrics/`).then((r) => r.json()),
        ])
            .then(([conf, metrics]) => {
                const c: AutomationConfig = {
                    automation_enabled: !!conf.automation_enabled,
                    automation_schedule: conf.automation_schedule ?? null,
                    promotion_metrics: Array.isArray(conf.promotion_metrics) ? conf.promotion_metrics : [],
                };
                setCfg(c);
                setAllowedMetrics(Array.isArray(metrics) ? metrics : []);
                // derive schedule UI mode/values
                if (c.automation_schedule && /^\d+$/.test(c.automation_schedule)) {
                    setMode('interval');
                    setIntervalSec(String(c.automation_schedule));
                    setCron('*/15 * * * *');
                } else {
                    setMode('cron');
                    setCron(c.automation_schedule || '*/15 * * * *');
                    setIntervalSec('900');
                }
            })
            .catch((e) => setError(String(e)))
            .finally(() => setLoading(false));
    }, [selectedTrId]);

    // current schedule string we will send to backend
    const scheduleOut = useMemo(() => {
        if (!cfg.automation_enabled) return null;
        if (mode === 'interval') {
            const s = (intervalSec || '').trim();
            return /^\d+$/.test(s) ? s : null;
        }
        const c = (cron || '').trim();
        return c.length ? c : null;
    }, [cfg.automation_enabled, mode, intervalSec, cron]);

    const save = async () => {
        if (!selectedTrId) return;
        setSavedOK(false);
        setSaving(true);
        setError(null);
        try {
            const body = {
                automation_enabled: cfg.automation_enabled,
                automation_schedule: scheduleOut, // may be null
                promotion_metrics: cfg.promotion_metrics,
            };
            const resp = await fetch(`${API}/trainings/${selectedTrId}/automation_config/`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `Save failed (HTTP ${resp.status})`);
            }
            setSavedOK(true);
            onChanged?.();
        } catch (e: any) {
            setError(e.message);
        } finally {
            setSaving(false);
        }
    };

    const runNow = async () => {
        if (!selectedTrId) return;
        setRunOK(false);
        setRunning(true);
        setError(null);
        try {
            const resp = await fetch(`${API}/trainings/${selectedTrId}/automation/run_now`, {
                method: 'POST',
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `Run now failed (HTTP ${resp.status})`);
            }
            setRunOK(true);
            onChanged?.();
        } catch (e: any) {
            setError(e.message);
        } finally {
            setRunning(false);
        }
    };

    return (
        <Drawer opened={opened} onClose={onClose} position="right" size="40%" title="Automation">
            <Group mb="xs" gap="xs">
                <IconRepeat size={18} />
                <Title order={4}>Pipeline Automation</Title>
                <Badge variant="light">beta</Badge>
            </Group>

            <Text size="sm" c="dimmed" mb="md">
                Automatically execute a pipeline (roots → preprocess → training) on a schedule, using the current
                active snapshots for all root datasources.
            </Text>

            <Divider my="xs" />

            <Select
                label="Training"
                placeholder="Select a training"
                value={selectedTrId || ''}
                data={trainings.map((t) => ({ value: t.id, label: t.name || t.id }))}
                onChange={setSelectedTrId}
                disabled={loading || !trainings.length}
                mb="md"
            />

            <Switch
                label="Enable automation"
                checked={cfg.automation_enabled}
                onChange={(e) => setCfg((c) => ({ ...c, automation_enabled: e.currentTarget.checked }))}
                mb="md"
            />

            <SegmentedControl
                value={mode}
                onChange={(v) => setMode(v as ScheduleMode)}
                data={[
                    { label: 'Interval (sec)', value: 'interval' },
                    { label: 'Cron', value: 'cron' },
                ]}
                disabled={!cfg.automation_enabled}
                mb="sm"
            />

            {mode === 'interval' ? (
                <TextInput
                    label="Interval seconds"
                    description="Number of seconds between runs (e.g., 900 for 15 minutes)."
                    value={intervalSec}
                    onChange={(e) => setIntervalSec(e.currentTarget.value)}
                    disabled={!cfg.automation_enabled}
                    mb="sm"
                />
            ) : (
                <TextInput
                    label="Cron schedule"
                    description="Standard crontab (e.g., */15 * * * *)."
                    value={cron}
                    onChange={(e) => setCron(e.currentTarget.value)}
                    disabled={!cfg.automation_enabled}
                    mb="sm"
                />
            )}

            <Divider my="sm" />

            <MultiSelect
                label="Promotion metrics (optional)"
                description="If set, the system will automatically keep the best execution per metric."
                data={allowedMetrics.map((m) => ({ value: m, label: m }))}
                value={cfg.promotion_metrics}
                onChange={(vals) => setCfg((c) => ({ ...c, promotion_metrics: vals }))}
                searchable
                nothingFoundMessage="No metrics"
                mb="md"
            />

            <Group mt="md" gap="sm">
                <Button onClick={save} loading={saving} disabled={!selectedTrId}>
                    Save
                </Button>
                <Button variant="light" leftSection={<IconRocket size={16} />} onClick={runNow} loading={running} disabled={!selectedTrId}>
                    Run now
                </Button>
            </Group>

            {savedOK && (
                <Notification mt="md" color="teal" withCloseButton={false} icon={<IconCheck />}>
                    Automation settings saved.
                </Notification>
            )}
            {runOK && (
                <Notification mt="md" color="indigo" withCloseButton={false} icon={<IconCheck />}>
                    Pipeline triggered — check training executions for progress.
                </Notification>
            )}
            {error && (
                <Notification mt="md" color="red" icon={<IconPlaystationX />}>
                    {error}
                </Notification>
            )}

            <Divider my="lg" />
            <Title order={6} mb="xs">
                Quick examples
            </Title>
            <Text size="sm" c="dimmed">
                <b>Interval:</b> 300 (every 5 minutes), 3600 (hourly)
                <br />
                <b>Cron:</b> <code>0 * * * *</code> (hourly), <code>0 3 * * *</code> (daily at 03:00),{' '}
                <code>*/15 * * * *</code> (every 15 minutes)
            </Text>
        </Drawer>
    );
}
