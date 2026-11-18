export function buildVirtualKeyboard() {
  return [
    "অ", "আ", "ই", "ঈ", "উ", "ঊ", "এ", "ও", "ক", "খ", "গ", "চ", "ছ", "জ", "ট", "ত", "প", "ব", "ম", "র", "্", "ে", "ি", "ী", "ো", "ৌ",
  ];
}

export function buildCharChips(text) {
  return Array.from(text || "");
}
