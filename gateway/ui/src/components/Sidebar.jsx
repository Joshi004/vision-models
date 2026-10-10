import { HistoryIcon } from './icons.jsx';

// Left-hand recipe picker: a vertical, grouped list on wider screens (md
// and up), collapsing into a single horizontally scrolling row of pills
// on narrow ones -- same markup and state either way, just different
// Tailwind classes per breakpoint, so there's exactly one mounted copy
// (no duplicated state to keep in sync).
export default function Sidebar({ groups, recipes, activeId, showingHistory, onSelectRecipe, onSelectHistory }) {
  function itemClass(isActive) {
    return `flex shrink-0 items-center gap-2 whitespace-nowrap rounded-md px-3 py-2 text-left text-sm transition-colors md:whitespace-normal ${
      isActive ? 'bg-accent text-white' : 'text-muted hover:bg-surface-2 hover:text-text'
    }`;
  }

  return (
    <nav className="flex shrink-0 flex-row items-center gap-1 overflow-x-auto border-b border-border bg-surface p-2 md:w-60 md:flex-col md:items-stretch md:overflow-visible md:border-b-0 md:border-r md:p-3">
      {groups.map((group, groupIndex) => (
        <div key={group.id} className="flex flex-row gap-1 md:mb-1 md:flex-col md:gap-0.5">
          <p
            className={`hidden px-2 pb-1 text-[11px] font-semibold uppercase tracking-wide text-muted md:block ${
              groupIndex === 0 ? 'pt-1' : 'pt-3'
            }`}
          >
            {group.label}
          </p>
          {recipes
            .filter((recipe) => recipe.group === group.id)
            .map((recipe) => {
              const RecipeIcon = recipe.icon;
              const isActive = !showingHistory && recipe.id === activeId;
              return (
                <button key={recipe.id} type="button" onClick={() => onSelectRecipe(recipe)} className={itemClass(isActive)}>
                  <RecipeIcon className="h-4 w-4 shrink-0" />
                  {recipe.label}
                </button>
              );
            })}
        </div>
      ))}
      <div className="flex flex-row gap-1 md:mt-2 md:flex-col md:border-t md:border-border md:pt-2">
        <button type="button" onClick={onSelectHistory} className={itemClass(showingHistory)}>
          <HistoryIcon className="h-4 w-4 shrink-0" />
          Job history
        </button>
      </div>
    </nav>
  );
}
