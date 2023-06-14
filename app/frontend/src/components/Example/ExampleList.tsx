import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "Which SQL operation is used to combine the content of two tables based on a shared column?",
        value: "Which SQL operation is used to combine the content of two tables based on a shared column?"
    },
    {
        text: "Which SQL clause can be used to copy all the rows from one table to a new table?",
        value: "Which SQL clause can be used to copy all the rows from one table to a new table?"
    },
    { text: "Which open-source database is a hybrid relational-object database?", value: "Which open-source database is a hybrid relational-object database?" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
