export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 py-16 px-6">
      <div className="max-w-2xl mx-auto prose prose-invert prose-sm">
        <h1 className="text-2xl font-bold text-white mb-2">Privacy Policy</h1>
        <p className="text-slate-500 text-xs mb-8">Last updated: March 2026</p>

        <h2 className="text-base font-semibold text-white mt-6 mb-2">What data we collect</h2>
        <p>We access your Xero accounting data via the Xero API, including bank transactions, chart of accounts, and bank statements. We store this data in a secure PostgreSQL database hosted on Railway (EU region).</p>

        <h2 className="text-base font-semibold text-white mt-6 mb-2">How we use your data</h2>
        <p>Transaction descriptions and amounts are sent to the Anthropic Claude API for AI categorisation and management letter generation. No personal contact data is sent to third-party AI services.</p>

        <h2 className="text-base font-semibold text-white mt-6 mb-2">Data storage</h2>
        <p>Your data is stored on Railway infrastructure in the EU. Xero OAuth tokens are stored in our database. Token encryption is planned before handling regulated client data. We do not sell or share your data with third parties beyond what is necessary to operate the service.</p>

        <h2 className="text-base font-semibold text-white mt-6 mb-2">Your rights under GDPR</h2>
        <p>As a data subject, you have the following rights, each of which is implemented in the service:</p>
        <ul className="list-disc pl-5 space-y-1 mt-2">
          <li><strong className="text-white">Right of access &amp; portability (Art. 15/20)</strong> — request a full export of all data held about your organisation via <code className="text-indigo-300">GET /api/v1/gdpr/export</code>. Returns structured JSON suitable for import into another system.</li>
          <li><strong className="text-white">Right to erasure (Art. 17)</strong> — request deletion of all your data via <code className="text-indigo-300">DELETE /api/v1/gdpr/erase</code>. All organisation rows are deleted in FK-safe order and your session is invalidated immediately.</li>
          <li><strong className="text-white">Right to explanation (Art. 22)</strong> — every AI categorisation decision stores its full reasoning, feature importances, and risk score in an immutable audit log, accessible via <code className="text-indigo-300">GET /api/v1/transactions/&#123;id&#125;/explanation</code>.</li>
          <li><strong className="text-white">Disconnect Xero</strong> — you can revoke access at any time from the Xero developer portal (<a href="https://developer.xero.com/myapps" className="text-indigo-400 hover:underline" target="_blank" rel="noreferrer">developer.xero.com/myapps</a>).</li>
        </ul>

        <h2 className="text-base font-semibold text-white mt-6 mb-2">Contact</h2>
        <p>
          Questions? Email us at{" "}
          <a href="mailto:rajarshin264@gmail.com" className="text-indigo-400 hover:underline">
            rajarshin264@gmail.com
          </a>
        </p>

        <div className="mt-10">
          <a href="/" className="text-sm text-indigo-400 hover:underline">← Back to home</a>
        </div>
      </div>
    </div>
  );
}
