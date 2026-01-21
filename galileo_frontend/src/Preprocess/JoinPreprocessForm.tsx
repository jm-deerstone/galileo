import React, { useState, useEffect } from 'react';
import {
    Button,
    Select,
    MultiSelect,
    TextInput,
    Text,
    Group,
    Notification,
    Loader,
    Center,
    Title,
    Drawer,
    Divider, ScrollArea, Table, Flex,
} from '@mantine/core';
import { IconArrowsJoin, IconCheck, IconX } from '@tabler/icons-react';
import { useDisclosure } from '@mantine/hooks';
import Editor from "@monaco-editor/react";

interface DataSource { id: string; name: string }
interface Snapshot   { id: string; created_at: string }

// Now includes the snapshot fields
type JoinParams = {
    left_datasource_id:  string;
    left_snapshot_id:    string;
    right_datasource_id: string;
    right_snapshot_id:   string;
    left_keys:           string[];
    right_keys:          string[];
    how:                 'inner' | 'left' | 'right' | 'outer' | 'custom';
    code:                string;
    suffixes:            string;
};

const codePlaceholder = `def join_step(left, right, params):
    """
    left, right   : pandas.DataFrame
    
    You can use the params object, which you are defining in the UI Form
    params        : dict with keys
        - left_keys, right_keys : lists of column names
        - suffixes              : "_l,_r" style string
    """
    # unpack
    lk = params.get("left_keys", [])
    rk = params.get("right_keys", [])
    
    # e.g. 'inner'|'left'|'right'|'outer'
    how = "inner"
    suffixes = tuple(params.get("suffixes", "_l,_r").split(","))

    # if you truly want a completely custom Python join,
    # you could instead exec() some user‐provided code here.
    # But for the built‐in, we just merge:
    return left.merge(
        right,
        left_on=lk,
        right_on=rk,
        how=how,
        suffixes=suffixes,
    )`


interface Props {
    /** Called whenever a new datasource (with snapshot) successfully created */
    onCreate(): void;
}

export function JoinPreprocessForm({ onCreate }: Props) {
    const [opened, { open, close }] = useDisclosure(false);
    const [datasources, setDatasources] = useState<DataSource[]>([]);
    const [leftSnaps,   setLeftSnaps]   = useState<Snapshot[]>([]);
    const [rightSnaps,  setRightSnaps]  = useState<Snapshot[]>([]);
    const [leftCols,    setLeftCols]    = useState<string[]>([]);
    const [rightCols,   setRightCols]   = useState<string[]>([]);

    // Initialize *all* JoinParams fields here
    const [params, setParams] = useState<JoinParams>({
        left_datasource_id:  '',
        left_snapshot_id:    '',
        right_datasource_id: '',
        right_snapshot_id:   '',
        left_keys:           [],
        right_keys:          [],
        how:                 'inner',
        code:                codePlaceholder,
        suffixes:            '_x,_y',
    });

    const [error,   setError]   = useState<string | null>(null);
    const [success, setSuccess] = useState(false);
    const [loading, setLoading] = useState(false);

    const [previewCols,    setPreviewCols]    = useState<string[]>([]);
    const [previewRows,    setPreviewRows]    = useState<string[][]>([]);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewError,   setPreviewError]   = useState<string | null>(null);

    // load datasources once when drawer opens
    useEffect(() => {
        if (!opened) return;
        fetch('http://localhost:8000/datasources/')
            .then(r => r.json())
            .then(setDatasources)
            .catch(console.error);
    }, [opened]);

    // load left snapshots + schema when left_datasource_id changes
    useEffect(() => {
        if (!params.left_datasource_id) return;
        fetch(`http://localhost:8000/datasources/${params.left_datasource_id}/with-snapshot/`)
            .then(r => r.json())
            .then(ds => {
                setLeftSnaps(ds.snapshots);
                const cols = JSON.parse(ds.schema_json || '{"columns":[]}').columns.map((c:any) => c.name);
                setLeftCols(cols);
            })
            .catch(console.error);
    }, [params.left_datasource_id]);

    // load right snapshots + schema when right_datasource_id changes
    useEffect(() => {
        if (!params.right_datasource_id) return;
        fetch(`http://localhost:8000/datasources/${params.right_datasource_id}/with-snapshot/`)
            .then(r => r.json())
            .then(ds => {
                setRightSnaps(ds.snapshots);
                const cols = JSON.parse(ds.schema_json || '{"columns":[]}').columns.map((c:any) => c.name);
                setRightCols(cols);
            })
            .catch(console.error);
    }, [params.right_datasource_id]);

    // helper to update any field
    function updateParam<K extends keyof JoinParams>(key: K, value: JoinParams[K]) {
        setParams(p => ({ ...p, [key]: value }));
    }

    // Preview handler
    async function handlePreview() {
        setPreviewError(null);
        setPreviewLoading(true);
        try {
            const body = {
                config: { steps: [{ op: 'join', params }] },
                snapshot_ids: {
                    [params.left_datasource_id]:  params.left_snapshot_id,
                    [params.right_datasource_id]: params.right_snapshot_id,
                }
            };
            const resp = await fetch('http://localhost:8000/preprocesses/preview/', {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify(body),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || resp.statusText);
            }
            const data = await resp.json();
            setPreviewCols(data.columns);
            setPreviewRows(data.rows);
        } catch (e: any) {
            setPreviewError(e.message);
        } finally {
            setPreviewLoading(false);
        }
    }

    // Create handler
    async function handleCreate() {
        setError(null);
        setSuccess(false);
        setLoading(true);
        try {
            const resp = await fetch('http://localhost:8000/preprocesses/', {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify({
                    name:       `Join ${params.left_datasource_id} & ${params.right_datasource_id}`,
                    parent_ids: [params.left_datasource_id, params.right_datasource_id],
                    config:     { steps: [{ op: 'join', params }] },
                }),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || resp.statusText);
            }
            setSuccess(true);
            onCreate()
        } catch (e: any) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <>
            <Button
                c={"#AAF1C9"}
                justify={"center"}
                variant="default"
                onClick={open}
                color={"#AAF1C9"}
                h={"auto"}
                w={"100%"}
                pt={"15px"}
                pb={"15px"}
            >
                <Flex direction={"column"} align={"center"}>
                    <IconArrowsJoin />
                    <Text>Join</Text>
                </Flex>
            </Button>

            <Drawer opened={opened} onClose={close} size="50%" title="Configure Join">
                <Title order={3} mb="md">Join Tables</Title>

                {/* LEFT */}
                <Select
                    label="Left Datasource"
                    data={datasources.map(d=>({ value:d.id, label:d.name }))}
                    value={params.left_datasource_id}
                    onChange={v=>updateParam('left_datasource_id', v!)}
                    mb="sm"
                />

                {/* RIGHT */}
                <Select
                    label="Right Datasource"
                    data={datasources.map(d=>({ value:d.id, label:d.name }))}
                    value={params.right_datasource_id}
                    onChange={v=>updateParam('right_datasource_id', v!)}
                    mb="sm"
                />

                <Divider my="md" />

                {/* OPTIONS */}
                <Select
                    label="Join Type"
                    data={['inner','left','right','outer', 'custom'].map(h=>({ value:h, label:h }))}
                    value={params.how}
                    onChange={v=>updateParam('how', v as any)}
                    mb="sm"
                />

                    <MultiSelect
                        label="Left Keys"
                        data={leftCols.map(c=>({ value:c, label:c }))}
                        value={params.left_keys}
                        onChange={v=>updateParam('left_keys', v)}
                        mb="md"
                        disabled={!leftCols.length}
                    />

                    <MultiSelect
                        label="Right Keys"
                        data={rightCols.map(c=>({ value:c, label:c }))}
                        value={params.right_keys}
                        onChange={v=>updateParam('right_keys', v)}
                        mb="md"
                        disabled={!rightCols.length}
                    />

                    <TextInput
                        label="Suffixes"
                        value={params.suffixes}
                        onChange={e=>updateParam('suffixes', e.currentTarget.value)}
                        mb="lg"
                    />


                {params.how === "custom" && <Editor
                    theme={"vs-dark"}
                    height="500px"
                    language="python"
                    value={params.code || ''}
                    onChange={(code) => updateParam('code', code!)}
                    options={{ minimap: { enabled: false } }}
                />}

                <Divider my="sm" />

                <Select
                    label="Left Snapshot (for preview)"
                    data={leftSnaps.map(s=>({
                        value:s.id,
                        label:new Date(s.created_at).toLocaleString()
                    }))}
                    value={params.left_snapshot_id}
                    onChange={v=>updateParam('left_snapshot_id', v!)}
                    mb="sm"
                    disabled={!params.left_datasource_id}
                />

                <Select
                    label="Right Snapshot (for preview)"
                    data={rightSnaps.map(s=>({
                        value:s.id,
                        label:new Date(s.created_at).toLocaleString()
                    }))}
                    value={params.right_snapshot_id}
                    onChange={v=>updateParam('right_snapshot_id', v!)}
                    mb="sm"
                    disabled={!params.right_datasource_id}
                />

                <Group mb="md">
                    <Button onClick={handlePreview} loading={previewLoading}>Preview</Button>
                    <Button color="green" onClick={handleCreate} loading={loading}>Create</Button>
                </Group>

                {/* PREVIEW TABLE */}
                {previewError && (
                    <Notification color="red" icon={<IconX />} onClose={()=>setPreviewError(null)} mb="sm">
                        {previewError}
                    </Notification>
                )}
                {previewLoading
                    ? <Center><Loader /></Center>
                    : previewCols.length > 0
                        ? <ScrollArea h={200}><Table striped>
                            <thead>
                            <tr>{previewCols.map(c => <th key={c}>{c}</th>)}</tr>
                            </thead>
                            <tbody>
                            {previewRows.map((r,i) => (
                                <tr key={i}>{r.map((cell,j)=><td key={j}>{cell}</td>)}</tr>
                            ))}
                            </tbody>
                        </Table></ScrollArea>
                        : <Text color="dimmed">Click “Preview” to see the first 5 rows.</Text>
                }

                {/* CREATE NOTIFICATIONS */}
                {error && (
                    <Notification color="red" icon={<IconX />} onClose={()=>setError(null)} mt="md">
                        {error}
                    </Notification>
                )}
                {success && (
                    <Notification color="teal" icon={<IconCheck />} onClose={()=>setSuccess(false)} mt="md">
                        Join preprocess created!
                    </Notification>
                )}
            </Drawer>
        </>
    );
}
