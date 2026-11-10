/** The Artifacts page's quick-create row: one small tile per thing a person can
 *  start here, an icon with its name under it. Each tile runs the same action
 *  the page already owns (new blank doc, import a file, new folder).
 *
 *  Each tile's `uiLocation` spread and label text sit on its own `Btn`: one id
 *  names one render site, and the index generator reads the label there. */
import { FilePlus, FileUp, FolderPlus, Loader2 } from 'lucide-react'
import { Btn } from '../components/ui'
import { uiLocation } from '../uiLocations/uiLocation'
import { i18nT } from '../i18n/t'

const TILE = 'flex-col items-center gap-1 w-[76px] px-1 py-2 border-transparent hover:border-border text-[12px] text-text'
const ICON_BOX = 'w-10 h-10 rounded-lg border border-border bg-bg-elevated flex items-center justify-center text-accent'
const LABEL = 'truncate max-w-full'

export function ArtifactsQuickCreate({ onNewDoc, newDocBusy, onImport, importBusy, onNewFolder }: {
  onNewDoc: () => void
  newDocBusy: boolean
  onImport: () => void
  importBusy: boolean
  onNewFolder: () => void
}) {
  return (
    <div role="group" aria-label={i18nT('pages.artifactsPage.create_new')} className="flex items-start gap-1 mb-4 -ml-1">
      <Btn
        onClick={onNewDoc}
        busy={newDocBusy}
        title={i18nT('pages.artifactsPage.start_a_new_blank_document_in_the_library')}
        className={TILE}
        {...uiLocation('artifacts.new')}
      >
        <span className={ICON_BOX}>{newDocBusy ? <Loader2 size={18} className="animate-spin" /> : <FilePlus size={18} />}</span>
        <span className={LABEL}>{i18nT('pages.artifactsPage.quick_doc')}</span>
      </Btn>
      <Btn
        onClick={onImport}
        busy={importBusy}
        title={i18nT('pages.artifactsPage.import_from_a_file')}
        className={TILE}
        {...uiLocation('artifacts.import')}
      >
        <span className={ICON_BOX}>{importBusy ? <Loader2 size={18} className="animate-spin" /> : <FileUp size={18} />}</span>
        <span className={LABEL}>{i18nT('pages.artifactsPage.quick_import')}</span>
      </Btn>
      <Btn
        onClick={onNewFolder}
        title={i18nT('pages.artifactsPage.create_a_folder_to_organize_your_artifacts')}
        className={TILE}
        {...uiLocation('artifacts.new-folder')}
      >
        <span className={ICON_BOX}><FolderPlus size={18} /></span>
        <span className={LABEL}>{i18nT('pages.artifactsPage.quick_folder')}</span>
      </Btn>
    </div>
  )
}
