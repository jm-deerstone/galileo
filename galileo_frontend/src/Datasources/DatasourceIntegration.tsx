import React, { useState, useEffect } from 'react';
import {
    Button,
    Center,
    Loader,
    Notification,
    Select,
    Badge,
    Code,
    Title,
    Stack, Flex,
} from '@mantine/core';
import { IconCheck, IconX } from '@tabler/icons-react';

interface SnapshotInfo {
    id: string;
    created_at: string;
}

interface DataSourceDetails {
    schema_json?: string;
    active_snapshot_id?: string;
    snapshots: SnapshotInfo[];
}

export function DataSourceIntegration({ id }: { id: string }) {
    const [details, setDetails] = useState<DataSourceDetails | null>(null);
    const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetch full datasource details (schema, snapshots, active_snapshot_id)
    const fetchDetails = async () => {
        setLoading(true);
        setError(null);
        try {
            const resp = await fetch(`http://localhost:8000/datasources/${id}/with-snapshot/`);
            if (!resp.ok) throw new Error(`Failed to load datasource (${resp.status})`);
            const ds: DataSourceDetails = await resp.json();
            setDetails(ds);
            // default to first snapshot if none selected yet
            if (!selectedSnapshotId && ds.snapshots.length > 0) {
                setSelectedSnapshotId(ds.snapshots[0].id);
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDetails();
    }, [id]);

    // Set the selected snapshot as active
    const handleSetActive = async () => {
        if (!selectedSnapshotId) return;
        setLoading(true);
        setError(null);
        try {
            const resp = await fetch(
                `http://localhost:8000/datasources/${id}/active_snapshot/`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ snapshot_id: selectedSnapshotId }),
                }
            );
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Could not set active snapshot');
            }
            // re-load details so active_snapshot_id is updated
            await fetchDetails();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Build the python snippet
    const buildSnippet = () => {
        if (!details || !details.schema_json) return '';
        const cols = JSON.parse(details.schema_json).columns.map((c: any) => c.name);
        const entries = cols
            .map((name: string) => `      "${name}": /* ${name}-value */`)
            .join(',\n');
        return `import requests

url = "http://localhost:8000/datasources/${id}/active_snapshot/rows"
payload = {
  "rows": [
    {
${entries}
    }
  ]
}

resp = requests.post(url, json=payload)
print(resp.json())
`;
    };

    const isActive = details?.active_snapshot_id === selectedSnapshotId;

    if (loading) {
        return (
            <Center style={{ height: 200 }}>
                <Loader />
            </Center>
        );
    }

    if (error) {
        return (
            <Notification color="red" icon={<IconX />} onClose={() => setError(null)}>
                {error}
            </Notification>
        );
    }

    if (!details) {
        return null;
    }

    if (!details?.active_snapshot_id){
        return <Flex direction={"column"} gap={"15px"}>
            <Select
                label="Snapshot"
                placeholder="Select snapshot"
                data={details.snapshots.map((s) => ({
                    value: s.id,
                    label: new Date(s.created_at).toLocaleString() + " " + s.id,
                }))}
                value={selectedSnapshotId}
                onChange={setSelectedSnapshotId}
            />

            {isActive ? (
                <Badge color="green">Active snapshot</Badge>
            ) : (
                <Flex>
                    <Button onClick={handleSetActive} disabled={!selectedSnapshotId}>
                        Set Active
                    </Button>
                </Flex>

            )}
            <Notification color="yellow" icon={<IconX />}>
                No active snapshot set on this datasource.
            </Notification>
        </Flex>
    }

    return (
        <Stack>
            <Select
                label="Snapshot"
                placeholder="Select snapshot"
                data={details.snapshots.map((s) => ({
                    value: s.id,
                    label: new Date(s.created_at).toLocaleString() + " " + s.id,
                }))}
                value={selectedSnapshotId}
                onChange={setSelectedSnapshotId}
            />

            {isActive ? (
                <Badge color="green">Active snapshot</Badge>
            ) : (
                <Button onClick={handleSetActive} disabled={!selectedSnapshotId}>
                    Set Active
                </Button>
            )}

            <Title order={4}>Python Snippet</Title>
            <Code block>{buildSnippet()}</Code>
            <Flex>
                <Button
                    leftSection={<IconCheck />}
                    onClick={() => navigator.clipboard.writeText(buildSnippet())}
                >
                    Copy to clipboard
                </Button>
            </Flex>

        </Stack>
    );
}
