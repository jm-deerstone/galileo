// src/DatasourceAnalysis.tsx
import React, { useState, useEffect } from 'react';
import {
    Card,
    Group,
    Select,
    MultiSelect,
    Loader,
    Center,
    Text,
    Title,
    Divider,
    Table,
    ScrollArea,
    Tabs,
    SimpleGrid,
    Slider,
    Progress,
} from '@mantine/core';
import { useDebouncedCallback } from '@mantine/hooks';

interface SnapshotInfo { id: string; created_at: string }
interface DataSourceDetails { snapshots: SnapshotInfo[]; schema_json?: string; }

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

interface ColumnSummary {
    column: string;
    type: 'numeric' | 'categorical' | 'date';
    missing: number;
    missing_pct: number;
    unique: number;
    stats: string;
}

export const DatasourceAnalysis = ({ id }: { id: string }) => {
    // — Snapshot selection —
    const [details, setDetails]           = useState<DataSourceDetails | null>(null);
    const [snaps, setSnaps]               = useState<SnapshotInfo[]>([]);
    const [selectedSnap, setSelectedSnap] = useState<string | null>(null);

    // — Server summaries & derived column lists —
    const [serverSummary, setServerSummary]     = useState<ColumnSummary[]>([]);
    const [numericCols, setNumericCols]         = useState<string[]>([]);
    const [categoricalCols, setCategoricalCols] = useState<string[]>([]);
    const [dateCols, setDateCols]               = useState<string[]>([]);

    // — Charts controls —
    const [histCol, setHistCol]     = useState<string | null>(null);
    const [bins, setBins]           = useState(10);
    const [scatterX, setScatterX]   = useState<string | null>(null);
    const [scatterY, setScatterY]   = useState<string | null>(null);
    const [pieCol, setPieCol]       = useState<string | null>(null);

    // — Matrix streaming state —
    const [pendingMatrixCols, setPendingMatrixCols] = useState<string[]>([]);
    const [matrixCols, setMatrixCols]               = useState<string[]>([]);
    const [matrixProgress, setMatrixProgress]       = useState(0);
    const [matrixImage, setMatrixImage]             = useState<string | null>(null);
    const [matrixLoading, setMatrixLoading]         = useState(false);

    const debouncedSetMatrixCols = useDebouncedCallback(
        (cols: string[]) => setMatrixCols(cols),
        { delay: 500, flushOnUnmount: true }
    );

    // — Build chart URLs —
    const histUrl = histCol && selectedSnap
        ? `${API_BASE}/datasources/${id}/snapshots/${selectedSnap}/histogram.png?col=${encodeURIComponent(histCol)}&bins=${bins}`
        : '';
    const scatterUrl = scatterX && scatterY && selectedSnap
        ? `${API_BASE}/datasources/${id}/snapshots/${selectedSnap}/scatter.png?x=${encodeURIComponent(scatterX)}&y=${encodeURIComponent(scatterY)}`
        : '';
    const pieUrl = pieCol && selectedSnap
        ? `${API_BASE}/datasources/${id}/snapshots/${selectedSnap}/pie.png?col=${encodeURIComponent(pieCol)}`
        : '';

    // — 1) Load datasource details & snapshots —
    useEffect(() => {
        fetch(`${API_BASE}/datasources/${id}/with-snapshot/`)
            .then(r => r.json())
            .then((ds: DataSourceDetails) => {
                setDetails(ds);
                setSnaps(ds.snapshots);
                if (ds.snapshots[0]) setSelectedSnap(ds.snapshots[0].id);
            })
            .catch(console.error);
    }, [id]);

    // — 2) Fetch full‐dataset summary from server on snapshot change —
    useEffect(() => {
        if (!selectedSnap) return;
        fetch(`${API_BASE}/datasources/${id}/snapshots/${selectedSnap}/summary/`)
            .then(r => {
                if (!r.ok) throw new Error(`Summary fetch failed (${r.status})`);
                return r.json();
            })
            .then((json: { summary: ColumnSummary[] }) => {
                const sum = json.summary;
                setServerSummary(sum);
                // derive column categories
                setNumericCols(sum.filter(c => c.type === 'numeric').map(c => c.column));
                setCategoricalCols(sum.filter(c => c.type === 'categorical').map(c => c.column));
                setDateCols(sum.filter(c => c.type === 'date').map(c => c.column));
                // initialize chart controls
                if (!histCol && sum.find(c => c.type === 'numeric')) {
                    setHistCol(sum.find(c => c.type === 'numeric')!.column);
                }
                const nums = sum.filter(c => c.type === 'numeric').map(c => c.column);
                if (!scatterX && nums.length > 1) {
                    setScatterX(nums[0]);
                    setScatterY(nums[1]);
                }
                if (!pieCol && sum.find(c => c.type === 'categorical')) {
                    setPieCol(sum.find(c => c.type === 'categorical')!.column);
                }
                // init matrix cols
                setPendingMatrixCols(nums.slice(0, Math.min(6, nums.length)));
                setMatrixCols(nums.slice(0, Math.min(6, nums.length)));
            })
            .catch(err => {
                console.error("Summary error:", err);
                setServerSummary([]);
            });
    }, [id, selectedSnap]);

    // — 3) Stream the matrix as SSE when matrixCols or snapshot changes —
    useEffect(() => {
        if (!selectedSnap || matrixCols.length < 2) return;
        setMatrixProgress(0);
        setMatrixImage(null);
        setMatrixLoading(true);

        // form comma‐separated cols param
        const colsParam = encodeURIComponent(matrixCols.join(','));
        const es = new EventSource(
            `${API_BASE}/datasources/${id}/snapshots/${selectedSnap}/pairwise_matrix_stream?cols=${colsParam}`
        );

        es.onmessage = e => {
            const { progress, image } = JSON.parse(e.data);
            setMatrixProgress(progress);
            setMatrixImage(`data:image/png;base64,${image}`);
        };
        es.addEventListener('done', () => {
            setMatrixLoading(false);
            es.close();
        });
        es.onerror = () => {
            setMatrixLoading(false);
            es.close();
        };

        return () => { es.close(); };
    }, [id, selectedSnap, matrixCols]);

    if (!details) {
        return (
            <Center style={{ height: 200 }}>
                <Loader />
            </Center>
        );
    }

    return (
        <Card shadow="sm" padding="md">
            {/* snapshot selector */}
            <Group mb="md">
                <Select
                    label="Snapshot"
                    data={snaps.map(s => ({
                        value: s.id,
                        label: new Date(s.created_at).toLocaleString(),
                    }))}
                    value={selectedSnap}
                    onChange={setSelectedSnap}
                    style={{ flex: 1 }}
                />
            </Group>

            <Tabs defaultValue="summary">
                <Tabs.List>
                    <Tabs.Tab value="summary">Summary</Tabs.Tab>
                    <Tabs.Tab value="charts">Charts</Tabs.Tab>
                    <Tabs.Tab value="matrix">Pairwise Matrix</Tabs.Tab>
                    <Tabs.Tab value="profile">Profile Report</Tabs.Tab> {/* <-- add this line */}
                </Tabs.List>

                {/* — Summary Tab — */}
                <Tabs.Panel value="summary" pt="md">
                    <Title order={4} mb="sm">Column Summary</Title>
                    <ScrollArea>
                        <Table striped highlightOnHover>
                            <Table.Thead>
                            <Table.Tr>
                                <Table.Th>Column</Table.Th>
                                <Table.Th>Type</Table.Th>
                                <Table.Th>Missing</Table.Th>
                                <Table.Th>% Missing</Table.Th>
                                <Table.Th>Unique</Table.Th>
                                <Table.Th>Stats</Table.Th>
                            </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                            {serverSummary.map(s => (
                                <Table.Tr key={s.column}>
                                    <Table.Td><Text size="sm">{s.column}</Text></Table.Td>
                                    <Table.Td><Text size="sm">{s.type}</Text></Table.Td>
                                    <Table.Td><Text size="sm">{s.missing}</Text></Table.Td>
                                    <Table.Td><Text size="sm">{s.missing_pct.toFixed(1)}%</Text></Table.Td>
                                    <Table.Td><Text size="sm">{s.unique}</Text></Table.Td>
                                    <Table.Td><Text size="sm">{s.stats}</Text></Table.Td>
                                </Table.Tr>
                            ))}
                            </Table.Tbody>
                        </Table>
                    </ScrollArea>
                </Tabs.Panel>

                {/* — Charts Tab — */}
                <Tabs.Panel value="charts" pt="md">
                    <SimpleGrid cols={3} spacing="md">
                        {/* Histogram */}
                        <div>
                            <Title order={5}>Histogram</Title>
                            <Group mb="xs">
                                <Select
                                    data={numericCols.map(c => ({ value: c, label: c }))}
                                    value={histCol}
                                    onChange={setHistCol}
                                    style={{ flex: 1 }}
                                />
                                <Slider
                                    value={bins}
                                    onChange={setBins}
                                    min={5}
                                    max={50}
                                    step={1}
                                    label={val => `${val} bins`}
                                    style={{ width: 200, marginLeft: 16 }}
                                />
                            </Group>
                            {histUrl
                                ? <img src={histUrl} alt="histogram" style={{ width: '100%' }} />
                                : <Text color="dimmed">Select a numeric column</Text>}
                        </div>

                        {/* Scatter */}
                        <div>
                            <Title order={5}>Scatter Plot</Title>
                            <Group mb="xs">
                                <Select
                                    data={numericCols.map(c => ({ value: c, label: c }))}
                                    value={scatterX}
                                    onChange={setScatterX}
                                />
                                <Select
                                    data={numericCols.map(c => ({ value: c, label: c }))}
                                    value={scatterY}
                                    onChange={setScatterY}
                                />
                            </Group>
                            {scatterUrl
                                ? <img src={scatterUrl} alt="scatter" style={{ width: '100%' }} />
                                : <Text color="dimmed">Pick X & Y</Text>}
                        </div>

                        {/* Pie */}
                        <div>
                            <Title order={5}>Category Pie</Title>
                            <Select
                                data={categoricalCols.map(c => ({ value: c, label: c }))}
                                value={pieCol}
                                onChange={setPieCol}
                                mb="xs"
                            />
                            {pieUrl
                                ? <img src={pieUrl} alt="pie" style={{ width: '100%' }} />
                                : <Text color="dimmed">Select a categorical column</Text>}
                        </div>
                    </SimpleGrid>
                </Tabs.Panel>

                {/* — Matrix Tab — */}
                <Tabs.Panel value="matrix" pt="md">
                    <Title order={5}>Pairwise Matrix</Title>
                    <MultiSelect
                        label="Columns"
                        data={numericCols.map(c => ({ value: c, label: c }))}
                        value={pendingMatrixCols}
                        onChange={vals => {
                            setPendingMatrixCols(vals);
                            debouncedSetMatrixCols(vals);
                        }}
                        placeholder="Select numeric columns…"
                        mb="sm"
                    />

                    <Progress value={matrixProgress * 100} mb="sm" />

                    {matrixImage
                        ? <img src={matrixImage} alt="pairwise matrix" style={{ width: '100%' }} />
                        : matrixLoading
                            ? <Center style={{ height: 200 }}><Loader /></Center>
                            : <Text color="dimmed">Pick two or more columns</Text>}
                </Tabs.Panel>
                <Tabs.Panel value="profile" pt="md">
                    <Title order={4} mb="sm">Pandas Profile Report</Title>
                    {selectedSnap ? (
                        <iframe
                            title="Pandas Profile Report"
                            src={`${API_BASE}/datasources/${id}/snapshots/${selectedSnap}/profile_report/`}
                            style={{
                                width: "100%",
                                minHeight: "800px",
                                border: "none",
                                background: "#222"
                            }}
                        />
                    ) : (
                        <Text color="dimmed">No snapshot selected.</Text>
                    )}
                </Tabs.Panel>
            </Tabs>
        </Card>
    );
};
