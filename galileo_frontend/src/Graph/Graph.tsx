import React, { useState, useEffect, useCallback } from 'react';
import {
    ReactFlow,
    Node,
    Edge,
    Controls,
    MiniMap,
    Background,
    MarkerType,
    Handle,
    Position,
} from '@xyflow/react';
import {
    Loader,
    Center,
    Text,
    Card,
    Flex,
} from '@mantine/core';
import '@xyflow/react/dist/style.css';
import { useDisclosure } from '@mantine/hooks';
import { DatasourceCenter } from '../Datasources/DatasourceCenter';
import { PreprocessCenter } from '../Preprocess/PreprocessCenter';
import { TrainingCenter } from '../Training/TrainingCenter';
import {
    IconFile3d,
    IconSettingsAutomation,
    IconArrowsJoin,
    IconManualGearbox,
    IconAutomaticGearbox,
    IconWorldWww, IconCloudUpload, IconBrain, IconSettings, IconDatabase
} from '@tabler/icons-react';
import { DeploymentForm } from "../Deployment/DeploymentForm";
import { DeploymentCenter } from "../Deployment/DeploymentCenter";
import { ApiClient } from "../api/client";

interface DataSource { id: string; name: string }
interface Preprocess {
    id: string;
    name: string;
    parent_ids: string[];
    child_id: string;
    config: { steps: { op: string; params: any }[] };
}
interface Training { id: string; name: string; datasource_id: string }
interface Deployment { id: string; training_id: string }


interface GraphProps {
    /** Whenever this integer changes, Graph will re‐fetch everything. */
    refreshKey: number;
}

export function Graph({ refreshKey }: GraphProps) {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const [selectedDsId, setSelectedDsId] = useState<string | null>(null);
    const [selectedPpId, setSelectedPpId] = useState<string | null>(null);
    const [selectedTrId, setSelectedTrId] = useState<string | null>(null);
    const [selectedDpId, setSelectedDpId] = useState<string | null>(null);

    const [drawerOpenDs, { open: openDrawerDs, close: closeDrawerDs }] = useDisclosure(false);
    const [drawerOpenPp, { open: openDrawerPp, close: closeDrawerPp }] = useDisclosure(false);
    const [drawerOpenTr, { open: openDrawerTr, close: closeDrawerTr }] = useDisclosure(false);
    const [drawerOpenDp, { open: openDrawerDp, close: closeDrawerDp }] = useDisclosure(false);

    const [onDelete, setOnDelete] = useState<number>(0)

    const handleDelete = () => {
        // simply bump the counter
        setOnDelete((k) => k + 1);
    };
    useEffect(() => {
        async function fetchGraph() {
            try {
                // 1) fetch all the pieces
                const [datasources, preprocesses, trainings, deployments] = await Promise.all([
                    ApiClient.getDatasources(),
                    ApiClient.getPreprocesses(),
                    ApiClient.getTrainings(),
                    ApiClient.getDeployments(),
                ]);

                // 2) build node & parent maps
                const nodesMap: Record<string, Node> = {};
                const parentsMap: Record<string, string[]> = {};

                datasources.forEach(ds => {
                    nodesMap[ds.id] = {
                        id: ds.id,
                        type: 'datasource',
                        position: { x: 0, y: 0 },
                        data: { label: ds.name },
                    };
                    parentsMap[ds.id] = [];
                });

                preprocesses.forEach(pp => {
                    const isJoin = pp.config.steps.some(s => s.op === 'join');
                    nodesMap[pp.id] = {
                        id: pp.id,
                        type: 'preprocess',
                        position: { x: 0, y: 0 },
                        data: { label: pp.name, isJoin },
                    };
                    parentsMap[pp.id] = [...pp.parent_ids];
                    parentsMap[pp.child_id] = parentsMap[pp.child_id] || [];
                    parentsMap[pp.child_id].push(pp.id);
                });

                trainings.forEach(tr => {
                    nodesMap[tr.id] = {
                        id: tr.id,
                        type: 'training',
                        position: { x: 0, y: 0 },
                        data: { label: tr.name },
                    };
                    parentsMap[tr.id] = [tr.datasource_id];
                });

                deployments.forEach(dp => {
                    const parentTr = trainings.find(t => t.id === dp.training_id);
                    nodesMap[dp.id] = {
                        id: dp.id,
                        type: 'deployment',
                        position: { x: 0, y: 0 },
                        data: { label: parentTr ? `${parentTr.name}` : 'Deployment' },
                    };
                    parentsMap[dp.id] = [dp.training_id];
                });

                // 3) build edges
                const dsToPp = preprocesses.flatMap(pp =>
                    pp.parent_ids.map((pid, i) => ({
                        id: `e-dspp-${pid}-${pp.id}-${i}`,
                        source: pid, target: pp.id,
                        type: 'smooth', animated: true,
                        markerEnd: { type: MarkerType.ArrowClosed },
                    }))
                );
                const ppToDs = preprocesses.map(pp => ({
                    id: `e-ppds-${pp.id}-${pp.child_id}`,
                    source: pp.id, target: pp.child_id,
                    type: 'smooth', animated: true,
                    markerEnd: { type: MarkerType.ArrowClosed },
                }));
                const dsToTr = trainings.map(tr => ({
                    id: `e-dstr-${tr.datasource_id}-${tr.id}`,
                    source: tr.datasource_id, target: tr.id,
                    type: 'smooth', animated: true,
                    markerEnd: { type: MarkerType.ArrowClosed },
                }));
                const trToDp = deployments.map(dp => ({
                    id: `e-trdp-${dp.training_id}-${dp.id}`,
                    source: dp.training_id, target: dp.id,
                    type: 'smooth', animated: true,
                    markerEnd: { type: MarkerType.ArrowClosed },
                }));
                const allEdges = [...dsToPp, ...ppToDs, ...dsToTr, ...trToDp];

                // 4) build undirected adjacency for components
                const adj: Record<string, Set<string>> = {};
                Object.keys(nodesMap).forEach(id => (adj[id] = new Set()));
                allEdges.forEach(e => {
                    adj[e.source].add(e.target);
                    adj[e.target].add(e.source);
                });

                // 5) find connected components
                const visited = new Set<string>();
                const components: string[][] = [];
                Object.keys(adj).forEach(start => {
                    if (visited.has(start)) return;
                    const stack = [start];
                    const comp: string[] = [];
                    while (stack.length) {
                        const u = stack.pop()!;
                        if (visited.has(u)) continue;
                        visited.add(u);
                        comp.push(u);
                        adj[u].forEach(v => { if (!visited.has(v)) stack.push(v) });
                    }
                    components.push(comp);
                });
                //6) layout each component
                const globalX = 200;
                const globalY = 100;
                let xOffset = 0;

                for (const comp of components) {
                    // a) compute layerMap…
                    const layerMap: Record<string, number> = {};
                    const dfsLayer = (id: string): number => {
                        if (layerMap[id] !== undefined) return layerMap[id];
                        const ps = (parentsMap[id] || []).filter(p => comp.includes(p));
                        const L = ps.length === 0
                            ? 0
                            : Math.max(...ps.map(dfsLayer)) + 1;
                        return (layerMap[id] = L);
                    };
                    comp.forEach(dfsLayer);

                    // b) group by layer
                    const byLayer: Record<number, Node[]> = {};
                    comp.forEach(id => {
                        const L = layerMap[id];
                        byLayer[L] = byLayer[L] || [];
                        byLayer[L].push(nodesMap[id]);
                    });

                    // c) assign positions, layer by layer
                    const layerIndices: Record<number, Record<string, number>> = {};

                    Object
                        .keys(byLayer)
                        .map(Number)
                        .sort((a, b) => a - b)
                        .forEach(L => {
                            const nodes = byLayer[L];

                            if (L === 0) {
                                // ── root layer: same as before ──
                                nodes.sort(/* typeOrder, label fallback */);
                                const idxMap: Record<string, number> = {};
                                nodes.forEach((node, idx) => {
                                    idxMap[node.id] = idx;
                                    node.position = { x: xOffset + idx * globalX, y: L * globalY };
                                });
                                layerIndices[L] = idxMap;

                            } else {
                                // ── L > 0: replace your old “spread” logic with this ──

                                // 1) collect parent indices from previous layer
                                const prevIdx = layerIndices[L - 1] || {};
                                const maxPrev = Object.values(prevIdx).length
                                    ? Math.max(...Object.values(prevIdx))
                                    : -1;

                                // 2) group children by sorted list of in-component parents
                                const groups: Record<string, string[]> = {};
                                nodes.forEach(node => {
                                    const key = (parentsMap[node.id] || [])
                                        .filter(pid => prevIdx[pid] !== undefined)
                                        .sort()
                                        .join(',');
                                    (groups[key] ||= []).push(node.id);
                                });

                                // 3) for each group, compute ideal columns, then shift to avoid overlap
                                let lastUsedCol = -1;
                                const idxMap: Record<string, number> = {};

                                for (const [key, ids] of Object.entries(groups)) {
                                    const N = ids.length;

                                    // compute “center” for non-orphan groups
                                    const base = key
                                        ? Math.round(
                                            key
                                                .split(',')
                                                .map(pid => prevIdx[pid])
                                                .reduce((a, b) => a + b, 0)
                                            / key.split(',').length
                                        )
                                        : null;

                                    // create ideal offsets around center (or orphans to the right of maxPrev)
                                    const offsets = ids.map((_, i) => i - Math.floor((N - 1) / 2));
                                    const initialCols = base !== null
                                        ? offsets.map(off => base + off)
                                        : ids.map((_, i) => maxPrev + 1 + i);

                                    // if this group’s leftmost would collide, shift right
                                    const groupMin = Math.min(...initialCols);
                                    const shift = Math.max(0, lastUsedCol + 1 - groupMin);
                                    const finalCols = initialCols.map(c => c + shift);

                                    // assign into idxMap and bump lastUsedCol
                                    ids.forEach((nid, i) => {
                                        idxMap[nid] = finalCols[i];
                                    });
                                    lastUsedCol = Math.max(lastUsedCol, ...finalCols);
                                }

                                // 4) position these nodes
                                nodes.forEach(node => {
                                    const col = idxMap[node.id];
                                    node.position = {
                                        x: xOffset + col * globalX,
                                        y: L * globalY,
                                    };
                                });
                                layerIndices[L] = idxMap;
                            }
                        });

                    // ── d) bump xOffset by component’s true width ──
                    const allColsSet = new Set<number>();
                    for (const idxMap of Object.values(layerIndices)) {
                        Object.values(idxMap).forEach(col => allColsSet.add(col));
                    }
                    const allColsArray = Array.from(allColsSet);   // ← convert to number[]
                    const maxCol = allColsArray.length
                        ? Math.max(...allColsArray)
                        : 0;
                    const compWidth = (maxCol + 1) * globalX;
                    xOffset += compWidth + 150;
                }

                // 7) commit
                setNodes(Object.values(nodesMap));
                setEdges(allEdges);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        }

        fetchGraph();
    }, [refreshKey, onDelete]);


    const onNodeClick = useCallback((_: any, node: Node) => {
        if (node.type === 'datasource') {
            setSelectedDsId(node.id);
            closeDrawerPp(); closeDrawerTr(); closeDrawerDp();
            openDrawerDs();
        }
        if (node.type === 'preprocess') {
            setSelectedPpId(node.id);
            closeDrawerDs(); closeDrawerTr(); closeDrawerDp();
            openDrawerPp();
        }
        if (node.type === 'training') {
            setSelectedTrId(node.id);
            closeDrawerDs(); closeDrawerPp(); closeDrawerDp();
            openDrawerTr();
        }
        if (node.type === 'deployment') {
            setSelectedDpId(node.id);
            closeDrawerDs(); closeDrawerPp(); closeDrawerTr();
            openDrawerDp();  // you’ll hook this up later
        }
    }, [
        openDrawerDs, closeDrawerDs,
        openDrawerPp, closeDrawerPp,
        openDrawerTr, closeDrawerTr,
        openDrawerDp, closeDrawerDp,
    ]);

    if (loading) return <Center style={{ height: '100vh' }}><Loader /></Center>;
    if (error) return <Center style={{ height: '100vh' }}><Text color="red">{error}</Text></Center>;

    return (
        <>
            <Card shadow="xl" withBorder style={{ width: '100%', height: '90vh' }}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    fitView
                    onNodeClick={onNodeClick}
                    nodeTypes={{
                        datasource: DatasourceNode,
                        preprocess: PreprocessNode,
                        training: TrainingNode,
                        deployment: DeploymentNode,
                    }}
                >
                    {/*
                    <Controls />
                    */}

                    <Background gap={12} size={1} />
                </ReactFlow>
            </Card>

            {selectedDsId && (
                <DatasourceCenter
                    id={selectedDsId} opened={drawerOpenDs} onClose={closeDrawerDs}
                />
            )}
            {selectedPpId && (
                <PreprocessCenter
                    id={selectedPpId} opened={drawerOpenPp} onClose={closeDrawerPp}
                />
            )}
            {selectedTrId && (
                <TrainingCenter
                    id={selectedTrId} opened={drawerOpenTr} onClose={closeDrawerTr} onDelete={handleDelete}
                />
            )}
            {selectedDpId && (
                <DeploymentCenter
                    deploymentId={selectedDpId}
                    opened={drawerOpenDp}
                    onClose={closeDrawerDp}
                />
            )}
            {/* You’ll wire up DeploymentCenter later */}
        </>
    );
}

// ———————————————————————————————————
// Node renderers

export const DatasourceNode = ({ data }: { data: { label: string } }) => {
    return (
        <>
            <Handle type="target" position={Position.Top} />


            <Flex direction="row" align="center" gap={4}>
                <Text
                    size="xs"
                    style={{ width: 100, wordBreak: 'break-word', visibility: "hidden" }}
                >
                    {data.label.slice(0, 20)}
                </Text>
                { // <IconDatabase color={"#E7F5FF"} size={70} />
                }
                <IconDatabase color={"#74A4E6"} size={70} />
                <Text
                    size="xs"
                    style={{ width: 100, wordBreak: 'break-word' }}
                >
                    {data.label.slice(0, 20)}
                </Text>
            </Flex>

            <Handle type="source" position={Position.Bottom} />
        </>
    );
};


export const PreprocessNode = ({ data }: { data: { label: string; isJoin?: boolean } }) => {
    // you can still vary the background if it’s a join node
    // const bg = data.isJoin ? '#AAF1C9' : '#FCF8E3';
    const bg = data.isJoin ? '#AAF1C9' : '#B8871F';

    return (
        <>
            <Handle type="target" position={Position.Top} />
            <Flex direction="row" align="center" gap={4}>
                <Text size="xs" style={{ width: 100, visibility: "hidden" }}>
                    {data.label.slice(0, 20)}
                </Text>
                {data.isJoin ? <IconArrowsJoin color={bg} size={70} /> : <IconSettings color={bg} size={70} />}

                <Text size="xs" style={{ width: 100 }}>
                    {data.label.slice(0, 20)}
                </Text>
            </Flex>

            <Handle type="source" position={Position.Bottom} />
        </>
    )
};

export const TrainingNode = ({ data }: { data: { label: string } }) => {

    return (
        <>
            <Handle type="target" position={Position.Top} />
            <Flex direction="row" align="center" gap={4}>
                <Text size="xs" style={{ width: 100, visibility: "hidden" }}>
                    {data.label.slice(0, 20)}
                </Text>
                <IconBrain color={"#D0B9F0"} size={70} />

                <Text size="xs" style={{ width: 100 }}>
                    {data.label.slice(0, 20)}
                </Text>
            </Flex>

            <Handle type="source" position={Position.Bottom} />
        </>
    );
};

export const DeploymentNode = ({ data }: { data: { label: string } }) => {

    return (
        <>
            <Handle type="target" position={Position.Top} />
            <Flex direction="row" align="center" gap={4}>
                <Text size="xs" style={{ width: 100, visibility: "hidden" }}>
                    {data.label.slice(0, 20)}
                </Text>
                <IconCloudUpload color={"#FFD8A8"} size={70} />

                <Text size="xs" style={{ width: 100 }}>
                    {data.label.slice(0, 20)}
                </Text>
            </Flex>

            <Handle type="source" position={Position.Bottom} />
        </>
    );
};