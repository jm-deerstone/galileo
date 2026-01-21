import React, {useEffect, useState} from 'react';
import {Center, Drawer, Loader, Tabs} from '@mantine/core';
import { IconMessageCircle, IconPhoto, IconSettings, IconChartHistogram,IconAi, IconNetwork } from '@tabler/icons-react';
import { DatasourceDetails } from './DatasourceDetails';
import {DatasourceAnalysis} from "./DatasourceAnalysis";
import {PreprocessCreateForm} from "../Preprocess/PreprocessCreateForm";
import {DataSourceIntegration} from "./DatasourceIntegration";


interface Props {
    id: string;
    opened: boolean;
    onClose: () => void;
}

export const DatasourceCenter: React.FC<Props> = ({ id, opened, onClose }) => {


    const [isRoot, setIsRoot] = useState(false);
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState("")
    // fetch datasource + schema + snapshots
    useEffect(() => {
        async function fetchDetails() {

            try {
                setLoading(true)
                // 2) determine root-ness
                const ppResp = await fetch('http://localhost:8000/preprocesses/');
                if (!ppResp.ok) throw new Error(`Could not load preprocesses`);
                const pps: { child_id: string }[] = await ppResp.json();
                console.log(id)
                console.log(pps)
                setIsRoot(!pps.some(pp => pp.child_id === id));
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        }
        fetchDetails();
    }, [id]);

    return (
        <Drawer
            opened={opened}
            onClose={onClose}
            title="Datasource Center"
            position="bottom"
            size="80%"
        >
            {loading && <Center><Loader></Loader></Center>}
            {!loading && <Tabs defaultValue="details">
                <Tabs.List>
                    <Tabs.Tab value="details" leftSection={<IconPhoto size={14} />}>
                        Details
                    </Tabs.Tab>
                    <Tabs.Tab value="analysis" leftSection={<IconChartHistogram size={14} />}>
                        Analysis
                    </Tabs.Tab>
                    {isRoot &&  <Tabs.Tab value="end2end" leftSection={<IconNetwork size={14} />}>
                        End2End Integration
                    </Tabs.Tab>}
                </Tabs.List>

                <Tabs.Panel value="details" pt="sm">
                    <DatasourceDetails id={id} />
                </Tabs.Panel>

                <Tabs.Panel value="analysis" pt="sm">
                    <DatasourceAnalysis id={id} />
                </Tabs.Panel>

                <Tabs.Panel value="end2end" pt="sm">
                    <DataSourceIntegration id={id}/>
                </Tabs.Panel>
            </Tabs>}

        </Drawer>
    );
};