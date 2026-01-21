// src/ModelMonitorDrawer.tsx
import React, { useState, useEffect } from 'react';
import {
    Drawer,
    Loader,
    Center,
    Notification,
    Title,
    Text,
    Group,
    Button,
    Table,
} from '@mantine/core';

interface MonitorRead {
    model_deployment_id: string;
    evaluated_on_snapshot: string;
    timestamp: string;
    metrics: Record<string, number | null>;
}

interface Props {
    mdId: string;
    opened: boolean;
    onClose(): void;
}

export function ModelMonitorDrawer({ mdId, opened, onClose }: Props) {
    const [loading, setLoading] = useState(false);
    const [error, setError]     = useState<string | null>(null);
    const [data, setData]       = useState<MonitorRead | null>(null);

    const runMonitor = async () => {
        setLoading(true);
        setError(null);
        setData(null);
        try {
            const resp = await fetch(
                `http://localhost:8000/model_deployments/${mdId}/monitor/`,
                { method: 'POST' }
            );
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || `Status ${resp.status}`);
            }
            const json: MonitorRead = await resp.json();
            setData(json);
        } catch (e: any) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!opened) {
            setData(null);
            setError(null);
            setLoading(false);
        }
    }, [opened]);

    return (
        <Drawer
            opened={opened}
            onClose={onClose}
            title="Monitor Model Performance"
            size="30%"
            position="right"
        >
            <Group mb="md">
                <Button onClick={runMonitor} loading={loading}>
                    Run Live Evaluation
                </Button>
            </Group>

            {loading && (
                <Center style={{ height: 100 }}>
                    <Loader />
                </Center>
            )}

            {error && (
                <Notification color="red" onClose={() => setError(null)}>
                    {error}
                </Notification>
            )}

            {data && (
                <>
                    <Title order={5} mb="xs">
                        Results ({new Date(data.timestamp).toLocaleString()})
                    </Title>
                    <Text size="sm" mb="sm">
                        Evaluated on snapshot {data.evaluated_on_snapshot.slice(0, 8)}…
                    </Text>
                    <Table striped>
                        <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        </thead>
                        <tbody>
                        {Object.entries(data.metrics).map(([k, v]) => (
                            <tr key={k}>
                                <td>{k}</td>
                                <td>{v == null ? '–' : v.toFixed?.(3) ?? v}</td>
                            </tr>
                        ))}
                        </tbody>
                    </Table>
                </>
            )}
        </Drawer>
    );
}
