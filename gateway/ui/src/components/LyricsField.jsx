import FieldLabel from './FieldLabel.jsx';
import Textarea from './ui/Textarea.jsx';
import Button from './ui/Button.jsx';

// A lyrics textarea with a button that inserts a ready-made list of section tags (ACE-Step and YuE2 both
// want the words under tags such as [Verse] / [Chorus]). `template` is the text to insert; with text
// already typed it is added below it, never replacing it.
export default function LyricsField({ model, field = 'lyrics', text = 'lyrics', value, onChange, template, rows = 10, required = false }) {
  function insertTemplate() {
    onChange(value.trim() === '' ? template : `${value.replace(/\s+$/, '')}\n\n${template}`);
  }

  return (
    <div className="flex flex-col gap-2">
      <FieldLabel model={model} field={field} text={text} required={required}>
        <Textarea value={value} onChange={(event) => onChange(event.target.value)} rows={rows} className="font-mono" />
      </FieldLabel>
      <div>
        <Button type="button" variant="secondary" size="sm" onClick={insertTemplate}>
          Insert section tags
        </Button>
      </div>
    </div>
  );
}
