
import React, { useState } from 'react';
import {
    TextInput,
    FileInput,
    Button,
    Group,
    Notification,
    Card,
    Text, Drawer, Flex
} from '@mantine/core';
import { IconCheck, IconDatabase, IconFile3d, IconX } from '@tabler/icons-react';
import { useDisclosure } from "@mantine/hooks";
import { ApiClient } from "../api/client";


interface Props {
    /** Called whenever a new datasource (with snapshot) successfully created */
    onCreate(): void;
}

export function CreateDatasourceWithSnapshotForm({ onCreate }: Props) {
    const [name, setName] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [opened, { open, close }] = useDisclosure(false);

    const handleSubmit = async (evt: React.FormEvent) => {
        evt.preventDefault();
        if (!name || !file) {
            setError('Both name and CSV file are required.');
            return;
        }
        setLoading(true);
        setError(null);
        setSuccess(null);

        const formData = new FormData();
        formData.append('name', name);
        formData.append('file', file);

        try {
            console.log('Submitting formData:', { name, fileName: file.name });
            const data = await ApiClient.createDatasourceWithSnapshot(name, file);
            console.log('Created datasource:', data);
            setSuccess(`Datasource "${data.name}" created!`);
            setName('');
            setFile(null);

            onCreate();

        } catch (err: unknown) {
            console.error('Submission error:', err);
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <Button
                c={"#E7F5FF"}
                justify={"center"}
                color={"#E7F5FF"}
                variant="default"
                onClick={open}
                h={"auto"}
                pt={"15px"}
                pb={"15px"}
                w={"100%"}
            >
                <Flex direction={"column"} align={"center"}>
                    <IconDatabase />
                    <Text>Datasource</Text>
                </Flex>
            </Button>

            <Drawer opened={opened} onClose={close} size={"50%"} title="Authentication">
                <form onSubmit={handleSubmit}>
                    <TextInput
                        label="Datasource Name"
                        placeholder="e.g. HousingData"
                        value={name}
                        onChange={(e) => setName(e.currentTarget.value)}
                        disabled={loading}
                        required
                        mb="md"
                    />

                    <FileInput
                        label="Initial CSV Snapshot"
                        placeholder="Upload .csv"
                        accept=".csv"
                        value={file}
                        onChange={setFile}
                        disabled={loading}
                        required
                        mb="md"
                    />

                    <Group mt="md">
                        <Button type="submit" loading={loading}>
                            Create & Upload
                        </Button>
                    </Group>
                </form>

                {error && (
                    <Notification
                        icon={<IconCheck size={18} />}
                        color="red"
                        onClose={() => setError(null)}
                        mt="md"
                    >
                        {error}
                    </Notification>
                )}
                {success && (
                    <Notification
                        icon={<IconCheck size={18} />}
                        color="teal"
                        onClose={() => setSuccess(null)}
                        mt="md"
                    >
                        {success}
                    </Notification>
                )}
            </Drawer>


        </>)
}