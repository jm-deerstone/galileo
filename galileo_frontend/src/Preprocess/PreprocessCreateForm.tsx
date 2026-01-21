// src/PreprocessCreateForm.tsx
import React, { useState, useEffect, ChangeEvent } from 'react';
import {
    Button,
    Card, Center,
    Code,
    Divider,
    Drawer,
    Flex,
    Group,
    Loader,
    MultiSelect,
    NumberInput,
    ScrollArea,
    Select,
    Stepper,
    Table,
    Text,
    TextInput,
    Title,
} from '@mantine/core';
import {IconPlus, IconTrash, IconCheck, IconSettings, IconArrowsJoin} from '@tabler/icons-react';
import { useDisclosure } from '@mantine/hooks';
import Editor from '@monaco-editor/react';
import type { DataSourceDetails } from '../Datasources/DatasourceDetails';

type OpKey =
    | 'rename_column'
    | 'drop_columns'
    | 'filter_rows'
    | 'filter_outliers'
    | 'impute_missing'
    | 'one_hot_encode'           // ← new
    | 'label_encode'      // ← new
    | 'scale_numeric'
    | 'log_transform'
    | 'extract_datetime_features'
    | 'remove_duplicates'
    | 'bin_numeric'
    | 'normalize_text'
    | 'cap_outliers'
    | 'custom_python';

interface PreprocessStep {
    op: OpKey;
    params: Record<string, any>;
}

const BOILERPLATE = `def step(df, params):
    # your custom pandas code here
    return df
`;

const OP_CONFIG: Record<
    OpKey,
    {
        label: string;
        description: string;
        inputs: Array<{
            name: string;
            type: 'text' | 'number' | 'select' | 'multiselect' | 'code';
            label: string;
            placeholder?: string;
            options?: { value: string; label: string }[];
        }>;
    }
> = {
    rename_column: {
        label: 'Rename Column',
        description: 'Rename one column to another.',
        inputs: [
            { name: 'from', type: 'text', label: 'Old name', placeholder: 'e.g. old_col' },
            { name: 'to', type: 'text', label: 'New name', placeholder: 'e.g. new_col' },
        ],
    },
    drop_columns: {
        label: 'Drop Columns',
        description: 'Remove one or more columns.',
        inputs: [{ name: 'columns', type: 'multiselect', label: 'Columns to drop' }],
    },
    filter_rows: {
        label: 'Filter Rows',
        description: 'Keep only rows matching a condition.',
        inputs: [
            { name: 'column', type: 'select', label: 'Column' },
            {
                name: 'operator',
                type: 'select',
                label: 'Operator',
                options: [
                    { value: '>', label: '>' },
                    { value: '<', label: '<' },
                    { value: '>=', label: '>=' },
                    { value: '<=', label: '<=' },
                    { value: '==', label: '==' },
                    { value: '!=', label: '!=' },
                ],
            },
            { name: 'value', type: 'text', label: 'Value' },
        ],
    },
    filter_outliers: {
        label: 'Filter Outliers',
        description: 'Remove rows with extreme values based on z-score or IQR.',
        inputs: [
            { name: 'column', type: 'select', label: 'Column' },
            {
                name: 'method',
                type: 'select',
                label: 'Method',
                options: [
                    { value: 'zscore', label: 'Z-score' },
                    { value: 'iqr', label: 'IQR' },
                ],
            },
            { name: 'threshold', type: 'number', label: 'Threshold', placeholder: 'e.g. 3' },
        ],
    },
    impute_missing: {
        label: 'Impute Missing',
        description: 'Fill missing values via mean, median, mode, etc.',
        inputs: [
            { name: 'column', type: 'select', label: 'Column' },
            {
                name: 'strategy',
                type: 'select',
                label: 'Strategy',
                options: [
                    { value: 'mean', label: 'Mean' },
                    { value: 'median', label: 'Median' },
                    { value: 'mode', label: 'Mode' },
                    { value: 'constant', label: 'Constant' },
                    { value: 'ffill', label: 'Forward fill' },
                    { value: 'bfill', label: 'Backward fill' },
                ],
            },
            { name: 'fill_value', type: 'text', label: 'Fill value (if constant)' },
        ],
    },
    // ——— ONE-HOT ENCODE —————
    one_hot_encode: {
        label: 'One-Hot Encode',
        description:
            'Turn a single categorical column into dummy columns. Supply the full list of categories ahead of time. Non listed Categories will be ignored and not converted into their own column.',
        inputs: [
            { name: 'column', type: 'select', label: 'Column to encode' },
            {
                name: 'categories',
                type: 'text',
                label: 'Categories (JSON array)',
                placeholder: 'e.g. ["A","B","C"]',
            },
            {
                name: 'drop_original',
                type: 'select',
                label: 'Drop original column?',
                options: [
                    { value: 'true', label: 'Yes' },
                    { value: 'false', label: 'No' },
                ],
            },
        ],
    },
    // ——— LABEL ENCODE —————
    label_encode: {
        label: 'Label Encode',
        description: 'Map each category to an integer; unseen categories will cause an error.',
        inputs: [{ name: 'column', type: 'select', label: 'Column to encode' }],
    },
    scale_numeric: {
        label: 'Scale Numeric',
        description: 'Min/max or standardize numeric columns.',
        inputs: [
            { name: 'columns', type: 'multiselect', label: 'Columns to scale' },
            {
                name: 'method',
                type: 'select',
                label: 'Method',
                options: [
                    { value: 'minmax', label: 'Min/Max' },
                    { value: 'standard', label: 'Standard' },
                ],
            },
        ],
    },
    log_transform: {
        label: 'Log Transform',
        description: 'Apply log(x + offset) to reduce skew.',
        inputs: [
            { name: 'columns', type: 'multiselect', label: 'Columns' },
            { name: 'offset', type: 'number', label: 'Offset', placeholder: 'e.g. 1e-6' },
        ],
    },
    extract_datetime_features: {
        label: 'Extract Date/Time Features',
        description: 'Pull year, month, day, etc. from a date column.',
        inputs: [
            { name: 'column', type: 'select', label: 'Date column' },
            {
                name: 'features',
                type: 'multiselect',
                label: 'Features',
                options: [
                    { value: 'year', label: 'Year' },
                    { value: 'month', label: 'Month' },
                    { value: 'day', label: 'Day' },
                    { value: 'hour', label: 'Hour' },
                    { value: 'weekday', label: 'Weekday' },
                ],
            },
        ],
    },
    remove_duplicates: {
        label: 'Remove Duplicates',
        description: 'Drop duplicate rows.',
        inputs: [
            { name: 'subset', type: 'multiselect', label: 'Subset columns (optional)' },
            {
                name: 'keep',
                type: 'select',
                label: 'Keep',
                options: [
                    { value: 'first', label: 'First' },
                    { value: 'last', label: 'Last' },
                    { value: 'false', label: 'None (drop all)' },
                ],
            },
        ],
    },
    bin_numeric: {
        label: 'Bin Numeric',
        description: 'Bucket a continuous column into bins.',
        inputs: [
            { name: 'column', type: 'select', label: 'Column to bin' },
            { name: 'bins', type: 'text', label: 'Bins (JSON array or int)' },
            { name: 'labels', type: 'text', label: 'Labels (JSON array, optional)' },
        ],
    },
    normalize_text: {
        label: 'Normalize Text',
        description: 'Lowercase and strip whitespace.',
        inputs: [
            { name: 'column', type: 'select', label: 'Text column' },
            {
                name: 'lowercase',
                type: 'select',
                label: 'Lowercase?',
                options: [
                    { value: 'true', label: 'Yes' },
                    { value: 'false', label: 'No' },
                ],
            },
            {
                name: 'strip',
                type: 'select',
                label: 'Strip whitespace?',
                options: [
                    { value: 'true', label: 'Yes' },
                    { value: 'false', label: 'No' },
                ],
            },
        ],
    },
    cap_outliers: {
        label: 'Cap Outliers',
        description: 'Clamp or winsorize extreme values.',
        inputs: [
            { name: 'column', type: 'select', label: 'Column' },
            {
                name: 'method',
                type: 'select',
                label: 'Method',
                options: [
                    { value: 'clip', label: 'Clip' },
                    { value: 'winsorize', label: 'Winsorize' },
                ],
            },
            { name: 'lower_pct', type: 'number', label: 'Lower percentile', placeholder: 'e.g. 0.01' },
            { name: 'upper_pct', type: 'number', label: 'Upper percentile', placeholder: 'e.g. 0.99' },
        ],
    },
    custom_python: {
        label: 'Custom Python',
        description: 'Write arbitrary pandas code.',
        inputs: [
            {
                name: 'code',
                type: 'code',
                label: 'Function body',
                placeholder: BOILERPLATE,
            },
        ],
    },
};

interface Props {
    /** Called whenever a new datasource (with snapshot) successfully created */
    onCreate(): void;
}

export function PreprocessCreateForm({ onCreate }: Props) {
    const [opened, { open, close }] = useDisclosure(false);

    // form state
    const [name, setName] = useState('');
    const [parentIds, setParentIds] = useState<string[]>([]);
    const [datasrcOptions, setDatasrcOptions] = useState<{ value: string; label: string }[]>([]);

    // snapshot & raw preview
    const [dsDetails, setDsDetails] = useState<DataSourceDetails | null>(null);
    const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(null);
    const [allLines, setAllLines] = useState<string[]>([]);
    const [rowCount, setRowCount] = useState(0);
    const [renderCount, setRenderCount] = useState(5);

    // transformation builder
    const [steps, setSteps] = useState<PreprocessStep[]>([
        { op: 'rename_column', params: { from: '', to: '' } },
    ]);

    // submission & navigation
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);
    const [active, setActive] = useState(0);
    const next = () => setActive((c) => Math.min(c + 1, 3));
    const prev = () => setActive((c) => Math.max(c - 1, 0));

    // live preview of transformed data
    const [previewCols, setPreviewCols] = useState<string[]>([]);
    const [previewRows, setPreviewRows] = useState<string[][]>([]);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewError, setPreviewError] = useState<string | null>(null);

    // load datasources list
    useEffect(() => {
        fetch('http://localhost:8000/datasources/')
            .then((r) => r.json())
            .then((list: any[]) =>
                setDatasrcOptions(list.map((d) => ({ value: d.id, label: d.name })))
            )
            .catch(console.error);
    }, [opened]);

    // when exactly one datasource is selected, fetch schema + RAW CSV preview
    useEffect(() => {
        if (parentIds.length !== 1) return;
        (async () => {
            try {
                const ds = await (
                    await fetch(
                        `http://localhost:8000/datasources/${parentIds[0]}/with-snapshot/`
                    )
                ).json();
                setDsDetails(ds);
                if (ds.snapshots.length) {
                    const sid = ds.snapshots[0].id;
                    setSelectedSnapshotId(sid);
                    const text = await fetch(
                        `http://localhost:8000/datasources/${parentIds[0]}/snapshots/${sid}/download`
                    ).then((r) => r.text());
                    const lines = text.split(/\r?\n/).filter((l) => l);
                    setAllLines(lines);
                    setRowCount(lines.length - 1);
                    setRenderCount(Math.min(5, lines.length - 1));
                }
            } catch (e) {
                console.error(e);
            }
        })();
    }, [parentIds]);

    // add / remove steps
    const addStep = () => setSteps((s) => [...s, { op: 'rename_column', params: {} }]);
    const removeStep = (i: number) => setSteps((s) => s.filter((_, idx) => idx !== i));

    // when a step param changes
    function handleParamChange(
        idx: number,
        field: string,
        value: any
    ) {
        const s = [...steps];
        s[idx].params = { ...s[idx].params, [field]: value };

        // auto-fill one-hot categories when column picked
        if (s[idx].op === 'one_hot_encode' && field === 'column') {
            const col = value as string;
            const headers = allLines[0]?.split(',');
            const ci = headers?.indexOf(col) ?? -1;
            if (ci >= 0) {
                const cats = Array.from(
                    new Set(allLines.slice(1).map((ln) => ln.split(',')[ci]).filter((v) => v))
                );
                s[idx].params.categories = JSON.stringify(cats);
            }
        }

        setSteps(s);
    }

    // fetch live transformed preview any time we enter Step 3
    useEffect(() => {
        if (active !== 2 || !selectedSnapshotId) return;
        setPreviewLoading(true);
        setPreviewError(null);
        fetch('http://localhost:8000/preprocesses/preview/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                snapshot_id: selectedSnapshotId,
                config: { steps },
            }),
        })
            .then(async (r) => {
                if (!r.ok) throw new Error(await r.text());
                return r.json();
            })
            .then((data: { columns: string[]; rows: string[][] }) => {
                setPreviewCols(data.columns);
                setPreviewRows(data.rows);
            })
            .catch((e) => setPreviewError(e.message))
            .finally(() => setPreviewLoading(false));
    }, [active, steps, selectedSnapshotId]);

    // final submit
    async function handleSubmit() {
        setSubmitting(true);
        setError(null);
        setSuccess(false);
        try {
            const resp = await fetch('http://localhost:8000/preprocesses/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, parent_ids: parentIds, config: { steps } }),
            });
            if (!resp.ok) {
                const e = await resp.json();
                throw new Error(e.detail || resp.statusText);
            }
            setSuccess(true);
            onCreate()
        } catch (e: any) {
            setError(e.message);
        } finally {
            setSubmitting(false);
        }
    }

    const columnOptions = dsDetails
        ? JSON.parse(dsDetails.schema_json || '{"columns":[]}').columns.map((c: any) => ({
            value: c.name,
            label: c.name,
        }))
        : [];

    return (
        <>
            <Button
                c={"#FCF8E3"}
                justify={"center"}
                variant="default"
                onClick={open}
                color={"#FCF8E3"}
                h={"auto"}
                w={"100%"}
                pt={"15px"}
                pb={"15px"}
            >
                <Flex direction={"column"} align={"center"}>
                    <IconSettings />
                    <Text>Preprocess</Text>
                </Flex>
            </Button>

            <Drawer opened={opened} onClose={close} size="80%" title="Create Preprocess">
                <Stepper active={active} onStepClick={setActive} mb="xl">
                    <Stepper.Step label="1" description="Choose data">
                        <Title order={4}>Step 1: Choose datasource</Title>
                        <TextInput
                            label="Name"
                            placeholder="CleanData"
                            value={name}
                            onChange={(e) => setName(e.currentTarget.value)}
                            mb="md"
                        />
                        <MultiSelect
                            label="Datasource (choose 1)"
                            data={datasrcOptions}
                            value={parentIds}
                            onChange={(v) => setParentIds(v.slice(0, 1))}
                            mb="md"
                        />
                        {dsDetails && (
                            <>
                                <Text mb="xs">Columns:</Text>
                                <ScrollArea h={120} mb="md">
                                    <Table striped>
                                        <thead>
                                        <tr>
                                            {JSON.parse(dsDetails.schema_json || '{"columns":[]}').columns.map(
                                                (c: any) => <th key={c.name}>{c.name}</th>
                                            )}
                                        </tr>
                                        </thead>
                                        <tbody>
                                        <tr>
                                            {JSON.parse(dsDetails.schema_json || '{"columns":[]}').columns.map(
                                                (c: any) => <td key={c.name}>{c.dtype}</td>
                                            )}
                                        </tr>
                                        </tbody>
                                    </Table>
                                </ScrollArea>
                            </>
                        )}
                    </Stepper.Step>

                    <Stepper.Step label="2" description="Build steps">
                        <Title order={4}>Step 2: Configure transforms</Title>
                        {steps.map((step, idx) => {
                            const cfg = OP_CONFIG[step.op];
                            return (
                                <Card key={idx} mb="md" withBorder padding="sm">
                                    <Group  mb="xs">
                                        <Text>
                                            Step {idx + 1}: {cfg.label}
                                        </Text>
                                        {steps.length > 1 && (
                                            <Button color="red" size="xs" variant="outline" onClick={() => removeStep(idx)}>
                                                <IconTrash size={14} />
                                            </Button>
                                        )}
                                    </Group>

                                    <Select
                                        label="Operation"
                                        data={Object.entries(OP_CONFIG).map(([k, c]) => ({ value: k, label: c.label }))}
                                        value={step.op}
                                        onChange={(v) => {
                                            const s = [...steps];
                                            if (v === 'custom_python') {
                                                s[idx] = { op: 'custom_python', params: { code: BOILERPLATE } };
                                            } else {
                                                s[idx] = { op: v as OpKey, params: {} };
                                            }
                                            setSteps(s);
                                        }}
                                        mb="xs"
                                    />

                                    <Text size="sm" color="dimmed" mb="sm">
                                        {cfg.description}
                                    </Text>

                                    {cfg.inputs.map((inp) => {
                                        const val = step.params[inp.name] ?? (inp.type === 'multiselect' ? [] : '');
                                        const opts = inp.options ?? columnOptions;

                                        switch (inp.type) {
                                            case 'text':
                                                return (
                                                    <TextInput
                                                        key={inp.name}
                                                        label={inp.label}
                                                        placeholder={inp.placeholder}
                                                        value={val as string}
                                                        onChange={(e) => handleParamChange(idx, inp.name, e.currentTarget.value)}
                                                        mb="xs"
                                                    />
                                                );
                                            case 'number':
                                                return (
                                                    <NumberInput
                                                        key={inp.name}
                                                        label={inp.label}
                                                        placeholder={inp.placeholder}
                                                        value={val as number}
                                                        onChange={(n) => handleParamChange(idx, inp.name, n!)}
                                                        mb="xs"
                                                    />
                                                );
                                            case 'select':
                                                return (
                                                    <Select
                                                        key={inp.name}
                                                        label={inp.label}
                                                        data={opts}
                                                        value={val as string}
                                                        onChange={(v) => handleParamChange(idx, inp.name, v!)}
                                                        mb="xs"
                                                    />
                                                );
                                            case 'multiselect':
                                                return (
                                                    <MultiSelect
                                                        key={inp.name}
                                                        label={inp.label}
                                                        data={opts}
                                                        value={val as string[]}
                                                        onChange={(v) => handleParamChange(idx, inp.name, v)}
                                                        mb="xs"
                                                    />
                                                );
                                            case 'code':
                                                return (
                                                    <Editor
                                                        key={inp.name}
                                                        theme="vs-dark"
                                                        height="200px"
                                                        language="python"
                                                        value={step.params.code}
                                                        onChange={(code) => handleParamChange(idx, 'code', code!)}
                                                        options={{ minimap: { enabled: false } }}
                                                    />
                                                );
                                        }
                                    })}
                                </Card>
                            );
                        })}

                        <Group  mb="md">
                            <Button leftSection={<IconPlus />} variant="subtle" onClick={addStep}>
                                Add Step
                            </Button>
                        </Group>
                    </Stepper.Step>

                    <Stepper.Step label="3" description="Preview & create">
                        <Title order={4}>Step 3: Preview</Title>

                        {previewLoading ? (
                            <Center><Loader/></Center>
                        ) : previewError ? (
                            <Text color="red">{previewError}</Text>
                        ) : (
                            <ScrollArea h={200} mb="md">
                                <Table striped>
                                    <thead>
                                    <tr>{previewCols.map((c) => <th key={c}>{c}</th>)}</tr>
                                    </thead>
                                    <tbody>
                                    {previewRows.map((row, ri) => (
                                        <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{cell}</td>)}</tr>
                                    ))}
                                    </tbody>
                                </Table>
                            </ScrollArea>
                        )}

                        <Divider my="md" />
                        <Group >
                            <Button
                                leftSection={<IconCheck />}
                                onClick={handleSubmit}
                                loading={submitting}
                            >
                                Create Preprocess
                            </Button>
                        </Group>

                        {error && <Text color="red" mt="md">{error}</Text>}
                        {success && <Text color="teal" mt="md">Created!</Text>}
                    </Stepper.Step>
                </Stepper>
            </Drawer>
        </>
    );
}