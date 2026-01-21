// src/DatasourceDetails.tsx
import React, { useState, useEffect } from 'react';
import {
    Table,
    Text,
    Title,
    Select,
    FileInput,
    Button,
    Group,
    Notification,
    Loader,
    Center,
    Divider,
    ScrollArea,
    Badge,
} from '@mantine/core';

interface DataSource {
    id: string;
    name: string;
}

interface SnapshotInfo {
    id: string;
    path: string;
    created_at: string;
    size_bytes: number;
}

export interface DataSourceDetails extends DataSource {
    schema_json?: string;
    active_snapshot_id?: string;
    snapshots: SnapshotInfo[];
}

export const DatasourceDetails = ({ id }: { id: string }) => {
    const MAX_DISPLAY = 5000;

    // details & preview state
    const [details, setDetails] = useState<DataSourceDetails | null>(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState<string | null>(null);

    // snapshot selection & upload
    const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(null);
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);

    // preview & metadata
    const [allLines, setAllLines] = useState<string[]>([]);
    const [renderCount, setRenderCount] = useState(5);
    const [rowCount, setRowCount] = useState(0);
    const [snapshotSize, setSnapshotSize] = useState<number | null>(null);
    const [previewLoading, setPreviewLoading] = useState(false);

    // fetch datasource + schema + snapshots
    useEffect(() => {
        async function fetchDetails() {
            setDetailLoading(true);
            try {
                const resp = await fetch(`http://localhost:8000/datasources/${id}/with-snapshot/`);
                if (!resp.ok) throw new Error(`Detail fetch failed: ${resp.status}`);
                const data: DataSourceDetails = await resp.json();
                setDetails(data);

                if (data.snapshots.length > 0) {
                    const firstId = data.snapshots[0].id;
                    setSelectedSnapshotId(firstId);
                    await fetchPreview(id, firstId);
                }
            } catch (err: any) {
                setDetailError(err.message);
            } finally {
                setDetailLoading(false);
            }
        }
        fetchDetails();
    }, [id]);

    // fetch CSV preview & metadata
    async function fetchPreview(dsId: string, snapId: string) {
        setPreviewLoading(true);
        try {
            // HEAD for size
            const headResp = await fetch(
                `http://localhost:8000/datasources/${dsId}/snapshots/${snapId}/download`,
                { method: 'HEAD' }
            );
            const sizeHeader = headResp.headers.get('content-length');
            setSnapshotSize(sizeHeader ? parseInt(sizeHeader, 10) : null);

            // GET for content
            const resp = await fetch(
                `http://localhost:8000/datasources/${dsId}/snapshots/${snapId}/download`
            );
            const text = await resp.text();
            const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
            setAllLines(lines);

            const dataLines = lines.slice(1);
            const totalRows = dataLines.length;
            setRowCount(totalRows);

            // initialize renderCount, capped at MAX_DISPLAY
            setRenderCount(Math.min(5, totalRows, MAX_DISPLAY));
        } catch (err) {
            console.error('Error fetching preview:', err);
        } finally {
            setPreviewLoading(false);
        }
    }

    // upload new snapshot
    const handleUpload = async () => {
        if (!id || !file) return;
        setUploading(true);
        setUploadError(null);
        try {
            const form = new FormData();
            form.append('file', file);
            const resp = await fetch(`http://localhost:8000/datasources/${id}/snapshots/`, {
                method: 'POST',
                body: form,
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Upload failed');
            }
            setUploadSuccess(true);

            // refresh details & preview
            const detailResp = await fetch(`http://localhost:8000/datasources/${id}/with-snapshot/`);
            const updated: DataSourceDetails = await detailResp.json();
            setDetails(updated);

            const newSnapId = updated.snapshots[0].id;
            setSelectedSnapshotId(newSnapId);
            await fetchPreview(id, newSnapId);
        } catch (err: any) {
            setUploadError(err.message);
        } finally {
            setUploading(false);
        }
    };

    if (detailLoading) {
        return (
            <Center style={{ height: 200 }}>
                <Loader />
            </Center>
        );
    }
    if (detailError) {
        return <Text color="red">Error: {detailError}</Text>;
    }
    if (!details) return null;

    // parse schema
    const columns = JSON.parse(details.schema_json || '{"columns":[]}').columns;

    // compute unique counts
    const uniqueCounts = columns.map((_: any, idx: number) => {
        const values = allLines.slice(1).map((line) => line.split(',')[idx]);
        return new Set(values).size;
    });

    // cap total rows for display
    const maxDisplayableRows = Math.min(rowCount, MAX_DISPLAY);
    const displayedRows = allLines.slice(1, 1 + renderCount);

    // compute selected snapshot size in KB
    const selectedSnapshot = details.snapshots.find((s) => s.id === selectedSnapshotId);
    const selectedSizeKB =
        selectedSnapshot != null ? (selectedSnapshot.size_bytes / 1024).toFixed(2) : null;

    return (
        <>
            <Title order={4} mb="sm">
                Columns & Types for Datasource
            </Title>
            <Table striped>
                <Table.Thead>
                <Table.Tr>
                    {columns.map((col: any) => (
                        <Table.Th key={col.name}>{col.name}</Table.Th>
                    ))}
                </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                <Table.Tr>
                    {columns.map((col: any) => (
                        <Table.Td key={col.name}>{col.dtype}</Table.Td>
                    ))}
                </Table.Tr>
                </Table.Tbody>
            </Table>

            <Divider my="md" />

            <Title order={4} mb="sm">
                Selected Snapshot
            </Title>
            <Text>{selectedSnapshot?.path}</Text>
            <Group align="flex-end" mb="md">
                <Select
                    label="Snapshot"
                    data={details.snapshots.map((s) => ({
                        value: s.id,
                        label: new Date(s.created_at).toLocaleString(),
                    }))}
                    value={selectedSnapshotId}
                    onChange={(val) => {
                        if (val) {
                            setSelectedSnapshotId(val);
                            fetchPreview(id, val);
                        }
                    }}
                    style={{ flex: 1 }}
                />
                <FileInput
                    placeholder="Upload CSV"
                    accept=".csv"
                    value={file}
                    onChange={setFile}
                    disabled={uploading}
                    style={{ flex: 1 }}
                />
                <Button onClick={handleUpload} disabled={!file} loading={uploading}>
                    Upload
                </Button>
            </Group>

            <Group mb="md">
                <Text>Rows: {rowCount}</Text>
                {selectedSizeKB != null && <Text>Size: {selectedSizeKB} KB</Text>}
            </Group>

            <Title order={5} mb="sm">
                Data Preview
            </Title>
            {previewLoading ? (
                <Loader />
            ) : (
                <>
                    <ScrollArea h={300}>
                        <Table striped>
                            <Table.Thead>
                            <Table.Tr>
                                {columns.map((col: any, idx: number) => (
                                    <Table.Th key={col.name}>
                                        {col.name}
                                        <br />
                                        <Text size="xs" color="dimmed">
                                            {uniqueCounts[idx]}
                                        </Text>
                                    </Table.Th>
                                ))}
                            </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                            {displayedRows.map((line, rowIndex) => {
                                const cells = line.split(',');
                                return (
                                    <Table.Tr key={rowIndex}>
                                        {cells.map((cell, ci) => (
                                            <Table.Td key={ci}>{cell}</Table.Td>
                                        ))}
                                    </Table.Tr>
                                );
                            })}
                            </Table.Tbody>
                        </Table>
                    </ScrollArea>

                    <Button.Group mt="md">
                        <Button
                            onClick={() => setRenderCount((c) => Math.min(c + 5, maxDisplayableRows))}
                            disabled={renderCount >= maxDisplayableRows}
                        >
                            Show more
                        </Button>
                        <Button
                            onClick={() => setRenderCount((c) => Math.max(c - 5, 5))}
                            disabled={renderCount <= 5}
                        >
                            Show less
                        </Button>
                        <Button
                            onClick={() => setRenderCount(maxDisplayableRows)}
                            disabled={renderCount >= maxDisplayableRows}
                        >
                            Show all ({maxDisplayableRows})
                        </Button>
                        <Button
                            onClick={() => setRenderCount(5)}
                            disabled={renderCount <= 5}
                        >
                            Hide all
                        </Button>
                    </Button.Group>
                </>
            )}

            {uploadError && (
                <Notification color="red" mt="md" onClose={() => setUploadError(null)}>
                    {uploadError}
                </Notification>
            )}
            {uploadSuccess && (
                <Notification color="teal" mt="md" onClose={() => setUploadSuccess(false)}>
                    Upload succeeded!
                </Notification>
            )}
        </>
    );
};
