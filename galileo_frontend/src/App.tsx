import React, {useState} from 'react';
import {AppShell, Text, Image, Button, Divider, Flex, Title, Combobox, Drawer} from "@mantine/core";
import {useDisclosure} from "@mantine/hooks";
import {Graph} from "./Graph/Graph";
import {CreateDatasourceWithSnapshotForm} from "./Datasources/CreateDatasourceWithSnapshotForm";
import {PreprocessCreateForm} from "./Preprocess/PreprocessCreateForm";
import {
    IconRepeat
} from '@tabler/icons-react';
import {JoinPreprocessForm} from "./Preprocess/JoinPreprocessForm";
import {TrainingForm} from "./Training/TrainingForm";

import logo from "./assets/Galileo_logo.png"
import Footer = Combobox.Footer;
import {DeploymentForm} from "./Deployment/DeploymentForm";

// NEW: automation drawer
import AutomationDrawer from './Automation/AutomationDrawer';

function App() {
    const [opened, { open, close }] = useDisclosure(false);

    // Automation drawer state
    const [autoOpen, { open: openAuto, close: closeAuto }] = useDisclosure(false);

    // This “refreshKey” will be passed down to Graph.
    // Whenever we bump this counter, Graph will re‐fetch.
    const [refreshKey, setRefreshKey] = useState(0);

    const handleDatasourceCreated = () => {
        setRefreshKey((k) => k + 1);
    };

    const handlePreproccesCreated = () => {
        setRefreshKey((k) => k + 1)
    }

    const handleTrainingCreated = () => {
        setRefreshKey((k) => k + 1)
    }

    const handleDeploymentCreated = () => {
        setRefreshKey((k) => k + 1)
    }

    // when automation saves or runs, we can also refresh the graph
    const handleAutomationChanged = () => {
        setRefreshKey((k) => k + 1);
    }

    return (
        <AppShell
            header={{ height: 60 }}
            navbar={{
                width: 300,
                breakpoint: 'sm',
            }}
            padding="md"
        >
            <AppShell.Header>
                <Flex h={"100%"} align={"center"} p={"15px"} gap={"15px"}>
                    <Title order={1}>Galileo</Title>
                </Flex>
            </AppShell.Header>

            <AppShell.Navbar p="md">
                <Flex direction={"column"} justify={"space-between"} h={"100%"}>
                    <Flex direction={"column"} gap={"15px"} >
                        <Flex direction={"row"} w={"100%"} gap={"15px"}>
                            <CreateDatasourceWithSnapshotForm onCreate={handleDatasourceCreated} />
                            {/* just so the button appears the way it does */}
                            <Button w={"100%"} style={{visibility: "hidden"}}></Button>
                        </Flex>

                        <Divider />

                        <Flex direction={"row"} gap={"15px"} w={"100%"}>
                            <PreprocessCreateForm onCreate={handlePreproccesCreated} />
                            <JoinPreprocessForm onCreate={handlePreproccesCreated} />
                        </Flex>

                        <Divider />

                        <Flex direction={"row"} gap={"15px"} w={"100%"}>
                            <TrainingForm onCreate={handleTrainingCreated} />
                            {/* just so the button appears the way it does */}
                            <Button w={"100%"} style={{visibility: "hidden"}}></Button>
                        </Flex>

                        <Divider />

                        <Flex direction={"row"} gap={"15px"} w={"100%"}>
                            <DeploymentForm onCreate={handleDeploymentCreated} />
                            {/* just so the button appears the way it does */}
                            <Button w={"100%"} style={{visibility: "hidden"}}></Button>
                        </Flex>

                        <Divider />

                        {/* NEW: Automation button opens drawer */}
                        <Flex direction={"row"} gap={"15px"} w={"100%"}>
                            <Button
                                c={"red"}
                                color={"red"}
                                variant={"default"}
                                justify={"center"}
                                h={"auto"}
                                w={"100%"}
                                pt={"15px"}
                                pb={"15px"}
                                onClick={openAuto}
                            >
                                <Flex direction={"column"} align={"center"}>
                                    <IconRepeat />
                                    <Text>Automation</Text>
                                </Flex>
                            </Button>
                            {/* just so the button appears the way it does */}
                            <Button w={"100%"} style={{visibility: "hidden"}}></Button>
                        </Flex>
                    </Flex>

                    <Divider />
                    <Button onClick={open}>Tasks</Button>
                    <Drawer opened={opened} onClose={close}>
                        {/* (left as-is) your Tasks content */}
                        <Title>Done</Title>
                        <Drawer opened={opened} onClose={close}>

                            <Title>Done</Title>
                            <Text c={"green"}>empty datasource {"->"} undefined error</Text>
                            <Text c={"green"}>Bin Preprocessing aint working</Text>
                            <Text c={"green"}>Filesize is wrong</Text>
                            <Text c={"green"}>Live Datensatz vergrößerung</Text>
                            <Text c={"green"}>fix python step view</Text>
                            <Text c={"green"}>Analysis panel rework</Text>
                            <Text c={"green"}>in Datasource summary of quality (e.g. balanced, unbalanced)</Text>
                            <Text c={"green"}>Wie weiß der client wie er seine daten preprocessen muss?</Text>
                            <Text c={"green"}>Wie kann man custom feature, target selection umsetzen? (e.g. more then one row...)</Text>
                            <Text c={"green"}>Frontend {"->"} too large csv!? what to do?</Text>
                            <Text c={"green"}>algorithms selection is weird (always linear!?)</Text>
                            <Text c={"green"}>Refresh graph on node created</Text>
                            <Text c={"green"}>Model Monitoring (active - trainedOn) -{'>'} preprocess -{'>'} execute models -{'>'} return target vs predicted</Text>
                            <Text c={"green"}>More algorithms</Text>
                            <Text c={"green"}>Training execution get created twice per run</Text>
                            <Text c={"green"}>Automatische hyperparamter selection</Text>
                            <Text c={"green"}>MLP REGRESSOR GIVES DIFFERENT OUTPUT MODELS FOR SAME INPUT - solution was fixed random state</Text>
                            <Text c={"green"}>After Tuning Algorithm finds params, save them into training object</Text>
                            <Text c={"green"}>after hyperparam object was saved, USE IT</Text>
                            <Text c={"green"}>make sure val is used in hyperparamtuning</Text>
                            <Text c={"green"}>Wie kann man custom Training split umsetzen?</Text>
                            <Text c={"green"}>Automation realization</Text>
                            <Title>Currently working on</Title>


                            <Text c={"yellow"}>Preprocess must always return same data schema (e.g. one hot encoding)</Text>
                            <Text c={"yellow"}>Create Preprocess panel rework</Text>


                            <Title>Not Done</Title>
                            <Text c={"red"}>Sub Menu for algorthims for overview for beginners -- maybe rather only description of algorithm</Text>



                            <Text c={"red"}>custom x y split via python script</Text>
                            <Text c={"red"}>custom Train / VALIDATION / Test split </Text>

                            <Text c={"red"}>Monitoring crashes when custom code preprocess is involved</Text>
                            <Text c={"red"}>Monitoring should use system generated inserted in date column</Text>

                            <Text c={"red"}>Automatische feature selection</Text>

                            <Text c={"red"}>Architektur refactoring</Text>


                            <Title>Nice to have maybe</Title>
                            <Text c={"blue"}>Other dataformats</Text>
                            <Text c={"blue"}>Monitoring as history graph. See how perfomance changes over time</Text>

                            <Text c={"blue"}>GPU if i find the time</Text>
                            <Text c={"blue"}>Custom dataformat by python adapter to pd.dataframe</Text>

                        </Drawer>
                    </Drawer>

                    <Image
                        src={logo}
                        alt="User portrait"
                        radius="md"
                    />
                </Flex>
            </AppShell.Navbar>

            <AppShell.Main>
                <Graph refreshKey={refreshKey}/>
            </AppShell.Main>

            {/* NEW: Automation drawer */}
            <AutomationDrawer opened={autoOpen} onClose={closeAuto} onChanged={handleAutomationChanged} />
        </AppShell>
    );
}

export default App;



