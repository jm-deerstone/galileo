// DeploymentForm.tsx
import React, { useState, useEffect } from 'react';
import {
    Button,
    Drawer,
    Select,
    Notification,
    Loader,
    Center,
    Title,
    Group, Flex, Text
} from '@mantine/core';
import {IconX, IconCheck, IconArrowsUpRight, IconWorldWww, IconCloudUpload} from '@tabler/icons-react';

interface TrainingRead {
    id: string;
    name: string;
}

interface Props {
    /** Called whenever a new datasource (with snapshot) successfully created */
    onCreate(): void;
}

export function DeploymentForm({ onCreate }: Props) {
    const [open, setOpen] = useState(false);
    const [trainings, setTrainings] = useState<TrainingRead[]>([]);
    const [selectedTraining, setSelectedTraining] = useState<string | null>(null);

    const [loading, setLoading] = useState(false);
    const [error, setError]       = useState<string | null>(null);
    const [success, setSuccess]   = useState<string | null>(null);

    // load trainings when drawer opens
    useEffect(() => {
        if (!open) return;
        setTrainings([]);
        setSelectedTraining(null);
        fetch('http://localhost:8000/trainings/')
            .then(r => {
                if (!r.ok) throw new Error(`Trainings ${r.status}`);
                return r.json();
            })
            .then((data: TrainingRead[]) => setTrainings(data))
            .catch(e => {
                console.error(e);
                setError('Failed to load trainings');
            });
    }, [open]);

    const handleCreate = async () => {
        if (!selectedTraining) {
            setError('Please select a training');
            return;
        }
        setLoading(true);
        setError(null);
        setSuccess(null);
        try {
            const resp = await fetch('http://localhost:8000/deployments/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ training_id: selectedTraining }),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Failed to create deployment');
            }
            const deployment = await resp.json();
            setSuccess(`Deployment created: ${deployment.id.slice(0, 8)}`);
            onCreate()
        } catch (e: any) {
            console.error(e);
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <Button
                c={"#FFD8A8"}
                color={"#FFD8A8"}
                variant={"default"}
                justify={"center"}
                onClick={() => setOpen(true)}
                h={"auto"}
                w={"100%"}
                pt={"15px"}
                pb={"15px"}
            >
                <Flex direction={"column"} align={"center"}>
                    <IconCloudUpload />
                    <Text>Deployment</Text>
                </Flex>
            </Button>

            <Drawer
                opened={open}
                onClose={() => setOpen(false)}
                title="Create Deployment"
                size="40%"
                position="right"
            >
                <Title order={4} mb="md">
                    Select Training to Deploy
                </Title>

                {trainings.length === 0 ? (
                    <Center style={{ height: 120 }}>
                        <Loader />
                    </Center>
                ) : (
                    <>
                        <Select
                            label="Training"
                            placeholder="Pick a training"
                            data={trainings.map(t => ({
                                value: t.id,
                                label: t.name,
                            }))}
                            value={selectedTraining}
                            onChange={setSelectedTraining}
                            mb="xl"
                        />

                        <Group>
                            <Button
                                onClick={handleCreate}
                                loading={loading}
                                disabled={!selectedTraining}
                            >
                                Create Deployment
                            </Button>

                            {error && (
                                <Notification
                                    color="red"
                                    onClose={() => setError(null)}
                                    icon={<IconX size={16} />}
                                >
                                    {error}
                                </Notification>
                            )}

                            {success && (
                                <Notification
                                    color="teal"
                                    onClose={() => setSuccess(null)}
                                    icon={<IconCheck size={16} />}
                                >
                                    {success}
                                </Notification>
                            )}
                        </Group>
                    </>
                )}
            </Drawer>
        </>
    );
}
