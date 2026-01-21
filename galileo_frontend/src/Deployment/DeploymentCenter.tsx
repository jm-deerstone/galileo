// src/DeploymentCenter.tsx
import React, { useState, useEffect } from 'react';
import {
    Button,
    Drawer,
    Select,
    Notification,
    Loader,
    Center,
    Title,
    Group,
    Text,
    Stack,
    Code,
    Box,
    Divider, Badge,
} from '@mantine/core';
import {IconX, IconCheck, IconCloudUpload, IconList, IconHeartRateMonitor} from '@tabler/icons-react';
import {ModelMonitorDrawer} from "../Monitor/ModelMonitorDrawer";

interface Deployment {
    id: string;
    training_id: string;
}

interface TrainingRead {
    id: string;
    config_json: {
        algorithm: string;
        params: Record<string, any>;
        window_spec: Record<string, any>;
        features?: string[];
        target?: string;
    };
    input_schema_json: {
        columns: { name: string; dtype: string; nullable?: boolean }[];
    };
}

interface ModelDeployment {
    id: string;
    training_execution_id: string;
    metric?: string;
    preprocess_details: Array<Record<string, any>>;
}

interface ExecutionRead {
    id: string;
    status: string;
}

interface Props {
    deploymentId: string;
    opened: boolean;
    onClose(): void;
}

export function DeploymentCenter({ deploymentId, opened, onClose }: Props) {
    const [deployment, setDeployment] = useState<Deployment | null>(null);
    const [training, setTraining] = useState<TrainingRead | null>(null);
    const [modelDeployments, setModelDeployments] = useState<ModelDeployment[]>([]);
    const [executions, setExecutions] = useState<ExecutionRead[]>([]);
    const [toDeploy, setToDeploy] = useState<string | null>(null);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const [monitorMdId, setMonitorMdId] = useState<string | null>(null);

    // 1) load deployment
    useEffect(() => {
        if (!opened) return;
        setError(null);
        setSuccess(null);
        setDeployment(null);
        setTraining(null);
        setModelDeployments([]);
        setExecutions([]);
        setToDeploy(null);

        fetch(`http://localhost:8000/deployments/${deploymentId}`)
            .then(r => {
                if (!r.ok) throw new Error(`Load failed (${r.status})`);
                return r.json();
            })
            .then((d: Deployment) => setDeployment(d))
            .catch(e => setError(e.message));
    }, [opened, deploymentId]);

    // 2a) fetch training
    useEffect(() => {
        if (!deployment) return;
        fetch(`http://localhost:8000/trainings/${deployment.training_id}`)
            .then(r => {
                if (!r.ok) throw new Error(`Train load failed (${r.status})`);
                return r.json();
            })
            .then((t: TrainingRead) => setTraining(t))
            .catch(e => setError(e.message));
    }, [deployment]);

    // 2b) fetch existing model deployments
    useEffect(() => {
        if (!deployment) return;
        fetch(`http://localhost:8000/deployments/${deployment.id}/model_deployments/`)
            .then(r => {
                if (!r.ok) throw new Error(`Models load failed (${r.status})`);
                return r.json();
            })
            .then((m: Array<{ id: string; training_execution_id: string }>) =>
                setModelDeployments(
                    m.map(md => ({
                        ...md,
                        preprocess_details: [],
                    }))
                )
            )
            .catch(e => setError(e.message));
    }, [deployment]);

    // 2c) fetch all successful executions
    useEffect(() => {
        if (!deployment) return;
        fetch(`http://localhost:8000/trainings/${deployment.training_id}/executions/`)
            .then(r => {
                if (!r.ok) throw new Error(`Executions load failed (${r.status})`);
                return r.json();
            })
            .then((all: ExecutionRead[]) => {
                setExecutions(all.filter(e => e.status === 'success'));
            })
            .catch(e => setError(e.message));
    }, [deployment]);

    // 3) fetch preprocess details for each model
    useEffect(() => {
        if (!deployment) return;
        const pending = modelDeployments.filter(md => md.preprocess_details.length === 0);
        if (!pending.length) return;

        Promise.all(
            pending.map(md =>
                fetch(
                    `http://localhost:8000/trainings/${deployment.training_id}/executions/${md.training_execution_id}/preprocess_steps/`
                )
                    .then(r => {
                        if (!r.ok) throw new Error(`Failed to load preprocess for ${md.id}`);
                        return r.json();
                    })
                    // endpoint returns Array<Record<string, any>>
                    .then((details: Array<Record<string, any>>) => ({
                        id: md.id,
                        details,
                    }))
            )
        )
            .then(results => {
                setModelDeployments(current =>
                    current.map(md => {
                        const found = results.find(r => r.id === md.id);
                        return found ? { ...md, preprocess_details: found.details } : md;
                    })
                );
            })
            .catch(e => setError(e.message));
    }, [deployment]);


    // 5) build code snippet
    function buildSnippet(md: ModelDeployment): string {
        const lines: string[] = [];

        // 5a) preprocessing comments
        if (md.preprocess_details.length) {
            lines.push('# --- Preprocessing Steps (in order) ---');
            md.preprocess_details.forEach((step, idx) => {
                const { op, ...rest } = step;
                lines.push(`# ${idx + 1}. ${op}: ${JSON.stringify(rest)}`);
            });
            lines.push('');
        }

        if (!training) {
            return lines.join('\n');
        }

        const url = `http://localhost:8000/model_deployments/${md.id}/predict`;
        const params = training.config_json.params || {};
        const windowSpec = training.config_json.window_spec;

        if (windowSpec) {
            console.log("window_spec codesnippet")
            // sliding-window snippet
            lines.push('import requests', '');
            lines.push(`url = "${url}"`, '');
            // each feature window
            windowSpec.features.forEach((f: any, i: number) => {
                lines.push(
                    `# ${i + 1}. ${f.name} window: rows [i - ${f.end_idx} .. i - ${f.start_idx}]`,
                    `${f.name}_values = [...]`,
                    ''
                );
            });
            lines.push(
                '# now flatten all windows into one list',
                `features = ${windowSpec.features
                    .map((f: any) => `${f.name}_values`)
                    .join(' + ')}`,
                'payload = { "features": features }',
                'resp = requests.post(url, json=payload)',
                '',
                'print(resp.json())'
            );
        } else {
            console.log("classic single-row snippet")
            //
            const features = training.config_json.features || [];
            lines.push('import requests', '');
            lines.push(`url = "${url}"`, '');
            lines.push('payload = { "features": {');
            features.forEach(name =>
                lines.push(`  "${name}": /* value for ${name} */,`)
            );
            lines.push('} }', '');
            lines.push('resp = requests.post(url, json=payload)');
            lines.push('print(resp.json())');
        }

        return lines.join('\n');
    }

    // exclude already-deployed executions
    const already = new Set(modelDeployments.map(md => md.training_execution_id));
    const candidates = executions
        .filter(e => !already.has(e.id))
        .map(e => ({ value: e.id, label: `${e.id.slice(0, 8)}…` }));

    return (
        <>
        <Drawer
            opened={opened}
            onClose={() => {
                setError(null);
                setSuccess(null);
                onClose();
            }}
            title="Manage Deployment"
            size="50%"
            position="right"
        >
            {!deployment ? (
                <Center style={{ height: 120 }}>
                    <Loader />
                </Center>
            ) : (
                <Stack>
                    <Group>
                        <Title order={4}>Deployment: {deployment.id.slice(0, 8)}…</Title>
                    </Group>

                    {error && (
                        <Notification color="red" icon={<IconX />} onClose={() => setError(null)}>
                            {error}
                        </Notification>
                    )}
                    {success && (
                        <Notification color="teal" icon={<IconCheck />} onClose={() => setSuccess(null)}>
                            {success}
                        </Notification>
                    )}

                    <Divider />

                    <Title order={5}>
                        <IconList size={16} /> Existing Models
                    </Title>
                    {modelDeployments.length === 0 ? (
                        <Text color="dimmed">No models deployed yet.</Text>
                    ) : (
                        modelDeployments.map(md => (
                            <Box key={md.id} p="sm" style={{ borderRadius: 4 }}>
                                <Text size="sm">
                                    Model: {md.id.slice(0, 8)}… | <Badge color={md.metric ? "teal" : "gray"}>
                                    {md.metric ? md.metric : "Manual"}
                                </Badge>
                                </Text>
                                <Text size="xs" color="dimmed" mb="xs">
                                    from training execution {md.training_execution_id}
                                </Text>
                                <Code block>{buildSnippet(md)}</Code>
                                <Button
                                    leftSection={<IconHeartRateMonitor />}
                                    mt="xs"
                                    size="xs"
                                    onClick={() => setMonitorMdId(md.id)}
                                >
                                    Monitor Performance
                                </Button>
                            </Box>
                        ))
                    )}

                    <Divider />

                </Stack>
            )}
        </Drawer>
    {monitorMdId && (
        <ModelMonitorDrawer
            mdId={monitorMdId}
            opened={!!monitorMdId}
            onClose={() => setMonitorMdId(null)}
        />
    )}
    </>

    );
}
