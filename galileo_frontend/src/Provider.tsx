import '@mantine/core/styles.css';
import '@mantine/charts/styles.css';
import {createTheme, MantineProvider} from "@mantine/core";
import App from "./App";


const theme = createTheme({
    /** Put your mantine theme override here */
});


export const Provider = () => {



    return <>
        <MantineProvider theme={theme} defaultColorScheme={"dark"} >
            <App />
        </MantineProvider>
    </>
}