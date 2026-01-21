export interface DataSource {
    id: string;
    name: string;
    schema_json?: string;
    active_snapshot_id?: string;
}

export interface Preprocess {
    id: string;
    name: string;
    parent_ids: string[];
    child_id: string;
    config: { steps: { op: string; params: any }[] };
}

export interface Training {
    id: string;
    name: string;
    datasource_id: string;
    config_json?: any;
    input_schema_json?: any;
}

export interface Deployment {
    id: string;
    training_id: string;
}

const BASE_URL = 'http://localhost:8000';

export class ApiClient {
    static async getDatasources(): Promise<DataSource[]> {
        const res = await fetch(`${BASE_URL}/datasources`);
        if (!res.ok) throw new Error(`Failed to fetch datasources: ${res.status}`);
        return res.json();
    }

    static async createDatasourceWithSnapshot(name: string, file: File): Promise<DataSource> {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('file', file);

        const res = await fetch(`${BASE_URL}/datasources/with-snapshot/`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(`Server returned ${res.status}: ${text}`);
        }
        return res.json();
    }

    static async getPreprocesses(): Promise<Preprocess[]> {
        const res = await fetch(`${BASE_URL}/preprocesses`);
        if (!res.ok) throw new Error(`Failed to fetch preprocesses: ${res.status}`);
        return res.json();
    }

    static async getTrainings(): Promise<Training[]> {
        const res = await fetch(`${BASE_URL}/trainings`);
        if (!res.ok) throw new Error(`Failed to fetch trainings: ${res.status}`);
        return res.json();
    }

    static async getDeployments(): Promise<Deployment[]> {
        const res = await fetch(`${BASE_URL}/deployments`);
        if (!res.ok) throw new Error(`Failed to fetch deployments: ${res.status}`);
        return res.json();
    }
}
