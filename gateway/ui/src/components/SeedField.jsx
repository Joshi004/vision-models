import FieldLabel from './FieldLabel.jsx';
import Input from './ui/Input.jsx';
import Button from './ui/Button.jsx';
import { DiceIcon } from './icons.jsx';

// A seed number input plus a "randomize" button -- every recipe form has
// one of these (see services/*/schemas.py's shared `seed` field).
export default function SeedField({ model, value, onChange }) {
  function randomize() {
    onChange(String(Math.floor(Math.random() * 1_000_000)));
  }

  return (
    <FieldLabel model={model} field="seed" text="seed">
      <div className="flex items-center gap-2">
        <Input type="number" value={value} onChange={(event) => onChange(event.target.value)} className="max-w-[10rem]" />
        <Button type="button" variant="secondary" size="sm" onClick={randomize} title="Randomize seed">
          <DiceIcon className="h-3.5 w-3.5" />
          Randomize
        </Button>
      </div>
    </FieldLabel>
  );
}
