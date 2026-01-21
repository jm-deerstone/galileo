// src/PreprocessCenter.tsx
import React, { useState, useEffect } from 'react';
import {
    Drawer,
    Tabs,
    Title,
    Text,
    Table,
    Select,
    Button,
    Group,
    Loader,
    Center,
    Notification,
    Divider,
    ScrollArea,
    Code,
} from '@mantine/core';
import { IconPhoto, IconCheck } from '@tabler/icons-react';

interface SnapshotInfo {
    id: string;
    created_at: string;
}

// Updated ExecutedRead to include `details`
interface ExecutedRead {
    id: string;
    input_snapshots: string[];
    output_snapshot: string;
    created_at: string;
    details: any[];             // <-- new details field
}

interface PreprocessRead {
    id: string;
    parent_ids: string[];
    config: { steps: { op: string; params: any }[] };
}

interface Props {
    id: string;
    opened: boolean;
    onClose(): void;
}

export const PreprocessCenter: React.FC<Props> = ({ id, opened, onClose }) => {
    const [pp, setPp] = useState<PreprocessRead | null>(null);
    const [executions, setExecutions] = useState<ExecutedRead[]>([]);
    const [snapshots, setSnapshots] = useState<Record<string, SnapshotInfo[]>>({});
    const [selectedSnaps, setSelectedSnaps] = useState<Record<string, string>>({});
    const [running, setRunning] = useState(false);
    const [runError, setRunError] = useState<string | null>(null);
    const [runSuccess, setRunSuccess] = useState<string | null>(null);

    // load preprocess & history & snapshots
    useEffect(() => {
        if (!opened) return;
        (async () => {
            setPp(null);
            setExecutions([]);
            setSnapshots({});
            setSelectedSnaps({});
            try {
                // 1) definition
                const ppRes = await fetch(`http://localhost:8000/preprocesses/${id}/`);
                const ppData: PreprocessRead = await ppRes.json();
                setPp(ppData);

                // 2) history
                const exRes = await fetch(`http://localhost:8000/preprocesses/${id}/executions/`);
                const exData: ExecutedRead[] = await exRes.json();
                setExecutions(exData);

                // 3) snapshots for each parent
                const map: Record<string, SnapshotInfo[]> = {};
                await Promise.all(
                    ppData.parent_ids.map(async (pid) => {
                        const dsRes = await fetch(`http://localhost:8000/datasources/${pid}/with-snapshot/`);
                        const dsJson = await dsRes.json();
                        map[pid] = dsJson.snapshots;
                    })
                );
                setSnapshots(map);
            } catch (e) {
                console.error(e);
            }
        })();
    }, [opened, id]);

    const handleRun = async () => {
        if (!pp) return;
        // ensure all parents have a selection
        for (const pid of pp.parent_ids) {
            if (!selectedSnaps[pid]) {
                setRunError(`Please select a snapshot for datasource ${pid}`);
                return;
            }
        }

        setRunning(true);
        setRunError(null);
        setRunSuccess(null);

        try {
            const payload =
                pp.parent_ids.length > 1
                    ? { snapshots: selectedSnaps }
                    : { snapshot_id: selectedSnaps[pp.parent_ids[0]] };

            const resp = await fetch(`http://localhost:8000/preprocesses/${id}/execute/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Execution failed');
            }
            setRunSuccess('Execution succeeded');
            // refresh history (including details)
            const exRes = await fetch(`http://localhost:8000/preprocesses/${id}/executions/`);
            setExecutions(await exRes.json());
        } catch (e: any) {
            setRunError(e.message);
        } finally {
            setRunning(false);
        }
    };

    return (
        <Drawer opened={opened} onClose={onClose} title="Preprocess Center" size="80%" position="bottom">
            {!pp ? (
                <Center style={{ height: 200 }}>
                    <Loader />
                </Center>
            ) : (
                <Tabs defaultValue="executions">
                    <Tabs.List>
                        <Tabs.Tab value="executions" leftSection={<IconPhoto size={14} />}>
                            History & Run
                        </Tabs.Tab>
                    </Tabs.List>

                    <Tabs.Panel value="executions" pt="md">
                        {/* Configured Steps */}
                        <Title order={4}>Configured Steps</Title>
                        <Table striped style={{ tableLayout: 'fixed', width: '100%' }}>
                            <Table.Thead>
                            <Table.Tr>
                                <Table.Th style={{ width: '20%' }}>Operation</Table.Th>
                                <Table.Th style={{ width: '80%' }}>Parameters</Table.Th>
                            </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                            {pp.config.steps.map((step, idx) => (
                                <Table.Tr key={idx}>
                                    <Table.Td><Text>{step.op}</Text></Table.Td>
                                    <Table.Td style={{ padding: 0 }}>
                                        {step.op === 'custom_python' ? (
                                            <ScrollArea style={{ maxHeight: 500, padding: '1rem' }}>
                                                <Code block style={{ whiteSpace: 'pre-wrap' }}>{step.params.code}</Code>
                                            </ScrollArea>
                                        ) : (
                                            <Code block style={{ maxHeight: 800, margin: 0, padding: '1rem' }}>
                                                {JSON.stringify(step.params, null, 2)}
                                            </Code>
                                        )}
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                            </Table.Tbody>
                        </Table>

                        {/* Run Now */}
                        <Divider my="md" />
                        <Title order={4}>Run Now</Title>
                        <Group mb="md" align="flex-end">
                            {pp.parent_ids.map((pid) => (
                                <Select
                                    key={pid}
                                    label={`Snapshot for ${pid}`}
                                    data={(snapshots[pid] || []).map((s) => ({
                                        value: s.id,
                                        label: new Date(s.created_at).toLocaleString(),
                                    }))}
                                    value={selectedSnaps[pid] || null}
                                    onChange={(v) =>
                                        setSelectedSnaps((prev) => ({ ...prev, [pid]: v! }))
                                    }
                                    style={{ flex: 1 }}
                                />
                            ))}
                            <Button leftSection={<IconCheck size={16} />} onClick={handleRun} loading={running}>
                                Execute
                            </Button>
                        </Group>

                        {runError && (
                            <Notification color="red" onClose={() => setRunError(null)} mb="sm">
                                {runError}
                            </Notification>
                        )}
                        {runSuccess && (
                            <Notification color="teal" onClose={() => setRunSuccess(null)} mb="sm">
                                {runSuccess}
                            </Notification>
                        )}

                        {/* Execution History */}
                        <Title order={4} mt="lg">Execution History</Title>
                        <ScrollArea>
                            <Table striped>
                                <Table.Thead>
                                <Table.Tr>
                                    <Table.Th>ID</Table.Th>
                                    <Table.Th>Inputs</Table.Th>
                                    <Table.Th>Output</Table.Th>
                                    <Table.Th>When</Table.Th>
                                    <Table.Th>Details</Table.Th> {/* new header */}
                                </Table.Tr>
                                </Table.Thead>
                                <Table.Tbody>
                                {executions.map((ex) => (
                                    <Table.Tr key={ex.id}>
                                        <Table.Td>{ex.id}</Table.Td>
                                        <Table.Td>{ex.input_snapshots.join(', ')}</Table.Td>
                                        <Table.Td>{ex.output_snapshot}</Table.Td>
                                        <Table.Td>{new Date(ex.created_at).toLocaleString()}</Table.Td>
                                        <Table.Td style={{ maxWidth: 300, padding: 0 }}>
                                            <ScrollArea style={{ maxHeight: 200, padding: '0.5rem' }}>
                                                <Code block style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>
                                                    {JSON.stringify(ex.details, null, 2)}
                                                </Code>
                                            </ScrollArea>
                                        </Table.Td>
                                    </Table.Tr>
                                ))}
                                </Table.Tbody>
                            </Table>
                        </ScrollArea>


                    </Tabs.Panel>
                </Tabs>
            )}
        </Drawer>
    );
};
